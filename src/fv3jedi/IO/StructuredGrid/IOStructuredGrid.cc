/*
 * (C) Copyright 2025- UCAR.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#include <netcdf.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <vector>

#include "atlas/functionspace.h"
#include "atlas/grid.h"

#include "oops/base/GeometryData.h"
#include "oops/base/Variables.h"
#include "oops/util/DateTime.h"
#include "oops/util/Duration.h"
#include "oops/util/Logger.h"
#include "oops/util/stringFunctions.h"
#include "oops/util/Timer.h"

#include "fv3jedi/FieldMetadata/FieldsMetadata.h"
#include "fv3jedi/Geometry/Geometry.h"
#include "fv3jedi/Increment/Increment.h"
#include "fv3jedi/IO/StructuredGrid/IOStructuredGrid.h"
#include "fv3jedi/State/State.h"

namespace fv3jedi {
// -------------------------------------------------------------------------------------------------
static IOMaker<IOStructuredGrid> makerIOStructuredGrid_("structured grid");
static IOMaker<IOStructuredGrid> makerIOAuxGrid_("auxgrid");
// -------------------------------------------------------------------------------------------------

// Build an Atlas grid Distribution that assigns all longitude columns in each latitude row
// to the same MPI rank, with rows divided as evenly as possible among nRanks ranks.
//
// This is a 1-D "latitude band" partitioner:
//   rank k owns ALL grid points in rows [k*nRows/nRanks, (k+1)*nRows/nRanks).
//
// Unlike atlas::grid::Partitioner("equal_regions") — which creates 2-D geographic patches
// where middle ranks share the same j_begin/j_end but each only owns a SUBSET of longitude
// columns — this distribution ensures that:
//   1. j_begin()/j_end() give the per-rank exclusive row range with no gaps between ranks.
//   2. i_begin(j)/i_end(j) = [0, nx(j)) for every owned row: ALL lon columns owned.
//   3. Every rank owns at least one row (as long as nRanks <= nRows), so GeometryData builds
//      its globalNodeTree_ on all ranks, satisfying the GlobalInterpolator requirement.
static atlas::grid::Distribution makeLatBandDistribution(const atlas::Grid & grid,
                                                          int nRanks) {
  const atlas::StructuredGrid sg(grid);
  const atlas::idx_t nRows = sg.ny();
  std::vector<int> partition;
  // grid.size() == sum of sg.nx(j) over all j, so this reserves exactly the right capacity.
  partition.reserve(static_cast<size_t>(grid.size()));
  for (atlas::idx_t j = 0; j < nRows; ++j) {
    // Clamp to nRanks-1 to guard against any edge-case integer-division result.
    const int rank = std::min(static_cast<int>((static_cast<atlas::idx_t>(j) * nRanks) / nRows),
                              nRanks - 1);
    for (atlas::idx_t i = 0; i < sg.nx(j); ++i) {
      partition.push_back(rank);
    }
  }
  return atlas::grid::Distribution(nRanks, static_cast<atlas::idx_t>(grid.size()),
                                   partition.data());
}

// -------------------------------------------------------------------------------------------------
IOStructuredGrid::IOStructuredGrid(const Geometry & geom, const Parameters_ & params)
  : IOBase(geom, params.toConfiguration()), interpolator_(), readInterpolator_(),
    params_(params), gridStr_(""),
    geom_(geom), writeFunctionSpace_(), readFunctionSpace_() {
  util::Timer timer(classname(), "IOStructuredGrid");
  oops::Log::trace() << classname() << " constructor starting" << std::endl;

  // Create the Atlas structured grid
  // --------------------------------
  // Create the string to determine the grid name for Atlas
  std::string outputGridType = params.outputGridType.value();

  // Convert the legacy gridtype to what Atlas expects
  if (outputGridType == "latlon") {
    outputGridType = "L" + std::to_string(4*(geom.npx()-1)) + "x" +
                     std::to_string(2*(geom.npy()-1)+1);
  } else if (outputGridType == "gaussian") {
    // Find best matching Gaussian grid
    outputGridType = "F" + std::to_string(geom.npy()-1);
  }

  // Assert that grid begins with either L or F
  if (outputGridType[0] == 'L') {
    gridStr_ = "latlon";
  } else if (outputGridType[0] == 'F') {
    gridStr_ = "gaussian";
  } else {
    // This code is only tested with latlon and regular Gaussian grids. With other grids the code
    // may run but with resulting files containing incorrect or jumbled data.
    ABORT("IOStructuredGrid: outputGridType must begin with L (latlon) or F (regular gaussian). ");
  }

  // Generate the Atlas grid object
  const atlas::Grid grid(outputGridType);

  // Make a custom serial distribution where all points live on rank 0
  // -----------------------------------------------------------------
  std::vector<int> zeros(grid.size(), 0);
  const atlas::grid::Distribution dist(geom.getComm().size(), grid.size(), zeros.data());

  // Create the configuration for the interpolation and populate with the communicator name
  // --------------------------------------------------------------------------------------
  eckit::LocalConfiguration atlas_conf;
  atlas_conf.set("mpi_comm", geom.getComm().name());

  // Structured grid function space (write)
  // ---------------------------------------
  writeFunctionSpace_.reset(new atlas::functionspace::StructuredColumns(grid, dist, atlas_conf));

  // Structured grid function space (read)
  // Uses a 1-D latitude-band distribution built by makeLatBandDistribution().
  // Each rank owns a contiguous band of complete latitude rows (all lon columns),
  // which satisfies two independent requirements:
  //   (a) GeometryData builds its globalNodeTree_ on ALL ranks because every rank
  //       owns > 0 points.  A serial (all-on-rank-0) distribution would leave ranks
  //       1+ with zero points, causing GeometryData to skip tree setup and
  //       GlobalInterpolator::closestTask() to assert !globalNodeTree_.empty().
  //   (b) The NC-file reading loop iterates i in [0, nLon) for each owned row j.
  //       With equal_regions (a 2-D geographic patch partitioner), middle ranks share
  //       the same j_begin/j_end but each owns only a lon slice.  Calling index(i,j)
  //       for an (i,j) not owned by this rank returns -1 or garbage.  Latitude bands
  //       guarantee i_begin(j)/i_end(j) = [0, nx(j)) for every owned row.
  // --------------------------------------------------------------------------------------
  const atlas::grid::Distribution readDist = makeLatBandDistribution(grid,
                                                                      geom.getComm().size());
  readFunctionSpace_.reset(new atlas::functionspace::StructuredColumns(grid, readDist,
                                                                       atlas_conf));

  // Create a GeometryData object for the write (cube-sphere → structured) interpolator
  // ------------------------------------------------------------------------------------
  oops::GeometryData geomData(geom.functionSpace(), geom.fields(), geom.levelsAreTopDown(),
                              geom.getComm());

  // Create a generic interpolator for converting to the structured grid (write path)
  // -------------------------------------------------------------------
  interpolator_.reset(new oops::GlobalInterpolator(params.toConfiguration(), geomData,
                                                   *writeFunctionSpace_,
                                                   geom.getComm()));

  // Create a GeometryData for the read (structured → cube-sphere) interpolator.
  // The source is the latitude-band readFunctionSpace_; an empty field set is sufficient
  // because the StructuredColumns function space provides its own coordinate information.
  // With latitude bands, GeometryData builds its globalNodeTree_ on all ranks, which is
  // required by GlobalInterpolator.
  // ------------------------------------------------------------------------------------
  atlas::FieldSet readGeomFields;
  oops::GeometryData readGeomData(*readFunctionSpace_, readGeomFields, geom.levelsAreTopDown(),
                                  geom.getComm());

  // Create an interpolator for converting from the structured grid to the cube-sphere
  // (read path: structured → cubed-sphere)
  // -------------------------------------------------------------------
  readInterpolator_.reset(new oops::GlobalInterpolator(params.toConfiguration(), readGeomData,
                                                       geom.functionSpace(),
                                                       geom.getComm()));
  oops::Log::trace() << classname() << " constructor done" << std::endl;
}
// -------------------------------------------------------------------------------------------------
IOStructuredGrid::~IOStructuredGrid() {
  util::Timer timer(classname(), "~IOStructuredGrid");
  oops::Log::trace() << classname() << " destructor starting" << std::endl;
  oops::Log::trace() << classname() << " destructor done" << std::endl;
}

// -------------------------------------------------------------------------------------------------

void IOStructuredGrid::read(State & x, const eckit::LocalConfiguration & fileionames,
                                const eckit::LocalConfiguration & fileioscaling) const {
  util::Timer timer(classname(), "read state");
  oops::Log::trace() << classname() << " read state starting" << std::endl;

  // Collect the field names requested by the State
  const oops::Variables & vars = x.variables();

  // original code below
  //const std::vector<std::string> fieldNames(vars.variables().begin(), vars.variables().end());

    // This is safer and more readable                                                                                                 
  const std::vector<std::string> fieldNames = vars.variables();

  oops::Log::trace() << classname() << " after load fieldNames " << std::endl;
  oops::Log::trace() << classname() << " fieldNames is " << fieldNames << std::endl;

  // Read fields from file(s) into the structured (readFunctionSpace_) Atlas FieldSet.
  // readFunctionSpace_ uses a latitude-band distribution so each MPI rank owns a contiguous
  // band of complete latitude rows, reads its slice from the file, and has a valid
  // GeometryData node tree — required by GlobalInterpolator::apply().
  // The valid time is also read from the first file and used to update x.validTime().
  // fileTime is initialised to x.validTime() as a fallback: if no time variable is found
  // in the file the State's existing valid time is preserved.
  atlas::FieldSet fieldsStructured;
  util::DateTime fileTime = x.validTime();  // fallback: unchanged when file has no time var
  this->readStructuredFields(fieldsStructured, fieldNames, x.validTime(), fileionames, &fileTime);
  x.validTime() = fileTime;

  // Interpolate from the structured grid to the cubed-sphere (all ranks participate).
  // Each rank contributes its latitude-band StructuredColumns rows to the interpolation.
  oops::Log::info() << classname() << " read state: applying structured→cube-sphere"
                    << " interpolation on grid '"
                    << readFunctionSpace_->grid().name() << "'" << std::endl;
  atlas::FieldSet fieldsCubeSphere;
  readInterpolator_->apply(fieldsStructured, fieldsCubeSphere);
  oops::Log::info() << classname() << " read state: interpolation done; populating State"
                    << std::endl;

  // Populate the State from the interpolated cubed-sphere FieldSet.
  x.fromFieldSet(fieldsCubeSphere);

  oops::Log::trace() << classname() << " read state done" << std::endl;
}

// -------------------------------------------------------------------------------------------------

void IOStructuredGrid::read(Increment & dx, const eckit::LocalConfiguration & fileionames,
                                const eckit::LocalConfiguration & fileioscaling) const {
  util::Timer timer(classname(), "read increment");
  oops::Log::trace() << classname() << " read increment starting" << std::endl;

  // Collect the field names requested by the Increment
  const oops::Variables & vars = dx.variables();
  const std::vector<std::string> fieldNames(vars.variables().begin(), vars.variables().end());

  // Read fields from file(s) into the structured (readFunctionSpace_) Atlas FieldSet.
  // readFunctionSpace_ uses a latitude-band distribution so each MPI rank owns a contiguous
  // band of complete latitude rows, reads its slice from the file, and has a valid
  // GeometryData node tree — required by GlobalInterpolator::apply().
  // The valid time is also read from the first file and used to update dx.validTime().
  // fileTime is initialised to dx.validTime() as a fallback: if no time variable is found
  // in the file the Increment's existing valid time is preserved.
  atlas::FieldSet fieldsStructured;
  util::DateTime fileTime = dx.validTime();  // fallback: unchanged when file has no time var
  this->readStructuredFields(fieldsStructured, fieldNames, dx.validTime(), fileionames, &fileTime);
  dx.validTime() = fileTime;

  // Interpolate from the structured grid to the cubed-sphere (all ranks participate).
  // Each rank contributes its latitude-band StructuredColumns rows to the interpolation.
  oops::Log::info() << classname() << " read increment: applying structured→cube-sphere"
                    << " interpolation on grid '"
                    << readFunctionSpace_->grid().name() << "'" << std::endl;
  atlas::FieldSet fieldsCubeSphere;
  readInterpolator_->apply(fieldsStructured, fieldsCubeSphere);
  oops::Log::info() << classname() << " read increment: interpolation done; populating Increment"
                    << std::endl;

  // Populate the Increment from the interpolated cubed-sphere FieldSet.
  dx.fromFieldSet(fieldsCubeSphere);

  oops::Log::trace() << classname() << " read increment done" << std::endl;
}

// -------------------------------------------------------------------------------------------------

template <typename T>
void IOStructuredGrid::interpAndWrite(const T & obj, const std::string & label,
                                      const eckit::LocalConfiguration & fileionames,
                                      const eckit::LocalConfiguration & fileioscaling) const {
  util::Timer timer(classname(), "write " + label);
  oops::Log::trace() << classname() << " write " << label << " starting" << std::endl;

  // Create field sets
  atlas::FieldSet fieldsCubeSphere;
  atlas::FieldSet fieldsGeographic;
  obj.toFieldSet(fieldsCubeSphere);

  // Apply interpolation
  interpolator_->apply(fieldsCubeSphere, fieldsGeographic);

  // Write to disk if rank 0
  if (geom_.getComm().rank() == 0) {
    this->writeStructuredFields(fieldsGeographic, obj.validTime(), fileionames, fileioscaling);
  }

  oops::Log::trace() << classname() << " write " << label << " done" << std::endl;
}

// -------------------------------------------------------------------------------------------------

void IOStructuredGrid::write(const State & x, const eckit::LocalConfiguration & fileionames,
                                const eckit::LocalConfiguration & fileioscaling) const {
  this->interpAndWrite(x, "state", fileionames, fileioscaling);
}

// -------------------------------------------------------------------------------------------------

void IOStructuredGrid::write(const Increment & dx, const eckit::LocalConfiguration & fileionames,
                             const eckit::LocalConfiguration & fileioscaling) const {
  this->interpAndWrite(dx, "increment", fileionames, fileioscaling);
}

// -------------------------------------------------------------------------------------------------

static inline void nc_rc(const int return_code, const std::string & operation) {
  // If there was a failure of the netCDF operation, abort with the error message
  if (return_code) {
    ABORT("IOStructuredGrid netCDF operation \'" + operation + "\' failed with error: "
          + nc_strerror(return_code));
  }
}

// -------------------------------------------------------------------------------------------------

// Read a global integer attribute from an open NetCDF file. If the attribute is absent, fall back
// to the length of the named dimension. Aborts on any other NetCDF error.
static int readGlobalIntAttrOrDimLen(int fileId, const std::string & attrName,
                                     const std::string & dimName) {
  int value;
  const int rc = nc_get_att_int(fileId, NC_GLOBAL, attrName.c_str(), &value);
  if (rc == NC_NOERR) {
    return value;
  }
  // Attribute not found — fall back to the length of the named dimension
  int dimId;
  nc_rc(nc_inq_dimid(fileId, dimName.c_str(), &dimId), "nc_inq_dimid " + dimName);
  size_t dimLen;
  nc_rc(nc_inq_dimlen(fileId, dimId, &dimLen), "nc_inq_dimlen " + dimName);
  return static_cast<int>(dimLen);
}

// -------------------------------------------------------------------------------------------------

// Given the number of longitudes (im) and latitudes (jm) from a NetCDF file, infer the Atlas
// regular Gaussian grid string "F<N>" whose ny() and nx(0) match. For a regular Gaussian grid
// F<N>: ny = 2*N and nx = 4*N. A small search window around N = jm/2 is used to tolerate any
// edge-case rounding.
static std::string inferAtlasGaussianGridString(int im, int jm) {
  const int N_guess = jm / 2;
  const int search_radius = 4;
  for (int delta = 0; delta <= search_radius; ++delta) {
    for (int sign : {1, -1}) {
      // At delta == 0 only test N_guess once (sign == 1 gives N_guess, skip sign == -1)
      if (delta == 0 && sign == -1) continue;
      const int N = N_guess + delta * sign;
      if (N <= 0) continue;
      const std::string candidate = "F" + std::to_string(N);
      try {
        const atlas::StructuredGrid g(candidate);
        if (g && g.ny() == jm && g.nx(0) == im) {
          return candidate;
        }
      } catch (...) {
        continue;
      }
    }
  }
  std::ostringstream oss;
  oss << "IOStructuredGrid::inferAtlasGaussianGridString: cannot find Atlas regular Gaussian "
      << "grid matching im=" << im << ", jm=" << jm
      << ". Tried N=" << (N_guess - search_radius) << " to N=" << (N_guess + search_radius) << ".";
  ABORT(oss.str());
  return "";  // unreachable, but required to satisfy the compiler
}

// -------------------------------------------------------------------------------------------------

/// Detect whether the latitude coordinate in an open NetCDF file is ordered north-to-south.
///
/// The function tries two candidate variable names in priority order:
///   1. "lat"        — common convention in structured-grid output files
///   2. latDimName   — dimension variable (e.g. "grid_yt" in UFS/GFS files)
///
/// For each candidate, the first and last latitude values are read (at longitude index 0
/// for 2-D coordinate arrays) and compared.  If lat[0] > lat[nLat-1] the file is considered
/// north-first (N→S).
///
/// If no recognisable latitude variable is found, north-first ordering is assumed (returns
/// true) so that packing is a no-op and the caller gets the same behaviour as before this
/// detection was introduced.
///
/// @param fileId      Open read-mode NetCDF file ID.
/// @param latDimName  Name of the latitude dimension (e.g. "grid_yt").
/// @param nLat        Number of latitude rows in the file.
/// @return            true  if lat[0] > lat[nLat-1]  (north-to-south, no j-flip needed).
///                    false if lat[0] < lat[nLat-1]  (south-to-north, j-flip needed).
static bool detectFileLatNorthFirst(int fileId, const std::string & latDimName, int nLat) {
  // Candidate variable names to check, in priority order
  const std::vector<std::string> candidates = {"lat", latDimName};

  for (const auto & varName : candidates) {
    int varId;
    if (nc_inq_varid(fileId, varName.c_str(), &varId) != NC_NOERR) continue;

    // Get number of dimensions for this variable
    int ndims;
    if (nc_inq_varndims(fileId, varId, &ndims) != NC_NOERR) continue;
    if (ndims < 1 || ndims > 2) continue;

    float lat0 = 0.0f;
    float latLast = 0.0f;

    if (ndims == 1) {
      // 1-D coordinate variable: lat(grid_yt)
      size_t start = 0, count = 1;
      if (nc_get_vara_float(fileId, varId, &start, &count, &lat0) != NC_NOERR) continue;
      start = static_cast<size_t>(nLat - 1);
      if (nc_get_vara_float(fileId, varId, &start, &count, &latLast) != NC_NOERR) continue;
    } else {
      // 2-D coordinate variable: lat(grid_yt, grid_xt) — read first longitude column
      size_t start2[2] = {0, 0};
      size_t count2[2] = {1, 1};
      if (nc_get_vara_float(fileId, varId, start2, count2, &lat0) != NC_NOERR) continue;
      start2[0] = static_cast<size_t>(nLat - 1);
      if (nc_get_vara_float(fileId, varId, start2, count2, &latLast) != NC_NOERR) continue;
    }

    const bool northFirst = (lat0 > latLast);
    oops::Log::trace() << "IOStructuredGrid: lat orientation from variable '" << varName
                       << "': lat[0]=" << lat0 << " lat[nLat-1]=" << latLast
                       << " -> northFirst=" << northFirst << std::endl;
    return northFirst;
  }

  // No recognisable lat variable found; assume north-first (no flip) and warn the user.
  oops::Log::warning() << "IOStructuredGrid: no latitude variable (tried 'lat' and '"
                       << latDimName << "') found in file; assuming north-to-south ordering "
                       << "(flipJ=false). Verify input file conventions." << std::endl;
  return true;
}

// -------------------------------------------------------------------------------------------------

/// Reads the valid time from an open NetCDF file.
///
/// Priority:
///   1. `time_iso` — 2-D character variable (dimensions [time, nchars] in C order, i.e.
///      (/nchars, time/) in Fortran order).  The ISO-8601 string at index `[0, :]` is read
///      and parsed directly into `util::DateTime`.
///   2. `time`     — numeric variable whose `units` attribute has the form
///      `"hours since YYYY-MM-DDTHH:MM:SS"`.  The base datetime is parsed from the units
///      string and the numeric offset (in hours) is added.
///
/// If neither variable is found the function returns `false` and `fileTime` is unchanged.
/// Logs the chosen method and the resulting datetime.
///
/// @param[in]  fileId    Open read-mode NetCDF file ID.
/// @param[out] fileTime  Receives the valid time on success.
/// @return               true on success, false if no recognised time variable is present.
static bool readValidTimeFromFile(int fileId, util::DateTime & fileTime) {
  // ------ 1. Try time_iso (ISO 8601 character variable) ------
  {
    int timeIsoId;
    if (nc_inq_varid(fileId, "time_iso", &timeIsoId) == NC_NOERR) {
      int ndims = 0;
      if (nc_inq_varndims(fileId, timeIsoId, &ndims) == NC_NOERR && ndims == 2) {
        int dimids[2];
        if (nc_inq_vardimid(fileId, timeIsoId, dimids) == NC_NOERR) {
          // C-order: dimids[0] = time dimension, dimids[1] = char dimension
          size_t charLen = 0;
          if (nc_inq_dimlen(fileId, dimids[1], &charLen) == NC_NOERR && charLen > 0) {
            size_t start[2] = {0, 0};
            size_t count[2] = {1, charLen};
            std::vector<char> buf(charLen + 1, '\0');
            if (nc_get_vara_text(fileId, timeIsoId, start, count, buf.data()) == NC_NOERR) {
              std::string isoStr(buf.data());
              // Trim at the first embedded null character
              const auto nullPos = isoStr.find('\0');
              if (nullPos != std::string::npos) isoStr.resize(nullPos);
              // Trim trailing whitespace
              while (!isoStr.empty() && (isoStr.back() == ' ' || isoStr.back() == '\t'))
                isoStr.pop_back();
              // Ensure a trailing 'Z' so util::DateTime can parse it
              if (!isoStr.empty() && isoStr.back() != 'Z') isoStr += "Z";
              if (!isoStr.empty()) {
                oops::Log::info() << "IOStructuredGrid: valid time from time_iso: "
                                  << isoStr << std::endl;
                fileTime = util::DateTime(isoStr);
                return true;
              }
            }
          }
        }
      }
    }
  }

  // ------ 2. Fall back to numeric time + "hours since ..." units ------
  {
    int timeId;
    if (nc_inq_varid(fileId, "time", &timeId) == NC_NOERR) {
      // Read the units attribute
      size_t attLen = 0;
      if (nc_inq_attlen(fileId, timeId, "units", &attLen) == NC_NOERR && attLen > 0) {
        std::vector<char> unitsVec(attLen + 1, '\0');
        if (nc_get_att_text(fileId, timeId, "units", unitsVec.data()) == NC_NOERR) {
          const std::string units(unitsVec.data(), attLen);
          const std::string prefix = "hours since ";
          if (units.substr(0, prefix.size()) == prefix) {
            // Parse the base datetime from the units string
            std::string baseDateStr = units.substr(prefix.size());
            // Trim trailing whitespace
            while (!baseDateStr.empty() &&
                   (baseDateStr.back() == ' ' || baseDateStr.back() == '\t'))
              baseDateStr.pop_back();
            // Ensure trailing 'Z'
            if (!baseDateStr.empty() && baseDateStr.back() != 'Z') baseDateStr += "Z";

            // Read the numeric time value at index 0 (hours offset from the base)
            double timeVal = 0.0;
            {
              size_t start = 0, count = 1;
              nc_get_vara_double(fileId, timeId, &start, &count, &timeVal);
            }

            // Compute the offset duration from hours → seconds.
            // The duration string requires a non-negative integer, so the sign of timeVal
            // is handled separately via the +/- operator below.
            const int64_t absSeconds =
                static_cast<int64_t>(std::round(std::abs(timeVal) * 3600.0));
            std::ostringstream durStr;
            durStr << "PT" << absSeconds << "S";
            const util::Duration offset(durStr.str());

            const util::DateTime base(baseDateStr);
            const util::DateTime result = (timeVal >= 0.0) ? base + offset : base - offset;

            oops::Log::info() << "IOStructuredGrid: valid time from time + units (units='"
                              << units << "', value=" << timeVal << "): "
                              << result << std::endl;
            fileTime = result;
            return true;
          }
        }
      }
    }
  }

  oops::Log::warning() << "IOStructuredGrid: no recognised time variable (time_iso or time) "
                       << "found in file; valid time not updated from file." << std::endl;
  return false;
}

// -------------------------------------------------------------------------------------------------

void IOStructuredGrid::writeStructuredFields(const atlas::FieldSet & fields,
                                             const util::DateTime & time,
                                             const eckit::LocalConfiguration & ioNames,
                                             const eckit::LocalConfiguration & ioScaling) const {
  // NetCDF IDs
  // ----------
  int fileId;

  // Dimension indices
  int latId;
  int lonId;
  int levId;
  int edgId;
  int forId;
  int timId;

  // Variable indices
  int fIv;
  std::map<std::string, int> fieldIvs;

  // Get ak/bk for writing
  // ---------------------
  std::vector<double> ak = geom_.ak();
  std::vector<double> bk = geom_.bk();

  // Get the name of the file and adjust with datetime
  // -------------------------------------------------
  std::string pathFile = params_.filename.value();

  // For backward compatibility add some things to the filename if not already present
  if (pathFile.find("%Y") == std::string::npos) {
    pathFile += "%Y%m%d_%H%M%Sz";
  }
  if (pathFile.find(".nc") == std::string::npos) {
    pathFile += ".nc4";
  }

  // Format the datetime string
  pathFile = time.formatString(pathFile);

  // Replace member number (ensemble applciaitons)
  util::stringfunctions::swapNameMember(params_.toConfiguration(), pathFile);

  // Create a file to write fields into
  // ----------------------------------
  nc_rc(nc_create(pathFile.c_str(), NC_CLOBBER | NC_NETCDF4, &fileId), "nc_create" + pathFile);

  // Create regular grid for determining lat/lon values
  // --------------------------------------------------
  const atlas::RegularGrid regGrid(writeFunctionSpace_->grid());

  // Define the dimensions in the file
  // ---------------------------------
  const int nLat = regGrid.ny();
  const int nLon = regGrid.nx();
  const int nLev = geom_.npz();
  const int nEdg = geom_.npz() + 1;
  const int nFor = 4;
  const int nTim = 1;

  nc_rc(nc_def_dim(fileId, params_.latName.value().c_str(), nLat, &latId), "nc_def_dim (lat)");
  nc_rc(nc_def_dim(fileId, params_.lonName.value().c_str(), nLon, &lonId), "nc_def_dim (lon)");
  nc_rc(nc_def_dim(fileId, params_.levName.value().c_str(), nLev, &levId), "nc_def_dim (lev)");
  nc_rc(nc_def_dim(fileId, params_.edgName.value().c_str(), nEdg, &edgId), "nc_def_dim (edg)");
  nc_rc(nc_def_dim(fileId, params_.forName.value().c_str(), nFor, &forId), "nc_def_dim (for)");
  nc_rc(nc_def_dim(fileId, params_.timName.value().c_str(), nTim, &timId), "nc_def_dim (tim)");

  // Define the dimensions variables in the file
  // -------------------------------------------
  std::vector<double> latArr(nLat);
  std::vector<double> lonArr(nLon);
  std::vector<int> levArr(nLev);
  std::vector<int> edgArr(nEdg);
  std::vector<int> forArr(nFor);
  std::vector<int> timArr(nTim);

  for (int i = 0; i < nLat; ++i) {
    latArr[i] = regGrid.y(nLat - 1 - i);
  }
  for (int i = 0; i < nLon; ++i) {
    lonArr[i] = regGrid.x(i);
  }
  for (int i = 0; i < nLev; ++i) {
    levArr[i] = i + 1;
  }
  for (int i = 0; i < nEdg; ++i) {
    edgArr[i] = i + 1;
  }
  for (int i = 0; i < nFor; ++i) {
    forArr[i] = i + 1;
  }
  for (int i = 0; i < nTim; ++i) {
    timArr[i] = i + 1;
  }

  // Write the dimension variables (and attributes) to the file
  // ----------------------------------------------------------
  nc_rc(nc_def_var(fileId, params_.latName.value().c_str(), NC_DOUBLE, 1, &latId, &fIv),
        "nc_def_var (lat)");
  nc_rc(nc_put_att_text(fileId, fIv, "units", strlen("degrees_north"), "degrees_north"),
        "nc_put_att_text (lat)");
  fieldIvs[params_.latName.value()] = fIv;

  nc_rc(nc_def_var(fileId, params_.lonName.value().c_str(), NC_DOUBLE, 1, &lonId, &fIv),
        "nc_def_var (lon)");
  nc_rc(nc_put_att_text(fileId, fIv, "units", strlen("degrees_east"), "degrees_east"),
        "nc_put_att_text (lon)");
  fieldIvs[params_.lonName.value()] = fIv;

  nc_rc(nc_def_var(fileId, params_.levName.value().c_str(), NC_INT, 1, &levId, &fIv),
        "nc_def_var (lev)");
  nc_rc(nc_put_att_text(fileId, fIv, "units", strlen("1"), "1"),
        "nc_put_att_text (lev)");
  fieldIvs[params_.levName.value()] = fIv;

  nc_rc(nc_def_var(fileId, params_.edgName.value().c_str(), NC_INT, 1, &edgId, &fIv),
        "nc_def_var (edg)");
  nc_rc(nc_put_att_text(fileId, fIv, "units", strlen("1"), "1"),
        "nc_put_att_text (edg)");
  fieldIvs[params_.edgName.value()] = fIv;

  nc_rc(nc_def_var(fileId, params_.forName.value().c_str(), NC_INT, 1, &forId, &fIv),
        "nc_def_var (for)");
  nc_rc(nc_put_att_text(fileId, fIv, "units", strlen("1"), "1"),
        "nc_put_att_text (for)");
  fieldIvs[params_.forName.value()] = fIv;

  nc_rc(nc_def_var(fileId, params_.timName.value().c_str(), NC_INT, 1, &timId, &fIv),
        "nc_def_var (tim)");
  nc_rc(nc_put_att_text(fileId, fIv, "units", strlen("1"), "1"),
        "nc_put_att_text (tim)");
  fieldIvs[params_.timName.value()] = fIv;

  // Define some categories of dimension IDs for fields
  // --------------------------------------------------
  std::map<int, std::vector<int>> fieldDims;
  fieldDims[nLev] = {timId, levId, latId, lonId};  // Fields at levels
  fieldDims[nEdg] = {timId, edgId, latId, lonId};  // Fields at edges
  fieldDims[4] = {timId, forId, latId, lonId};     // Fields at four levels
  fieldDims[1] = {timId, latId, lonId};            // Fields at surface
  fieldDims[0] = {timId, latId, lonId};            // Fields at surface

  // Set float precision for fields
  // ------------------------------
  const int floatPrecision = params_.floatPrecision.value();
  const int ncPrec = (floatPrecision == 4) ? NC_FLOAT : NC_DOUBLE;

  // Define all the fields that will be written
  // ------------------------------------------
  for (auto& field : fields) {
    // Get number of levels for this field
    const int nLevField = field.shape(1);

    // Get dimensions for this field from map
    auto it = fieldDims.find(nLevField);
    if (it == fieldDims.end()) {
      std::ostringstream oss;
      oss << "IOStructuredGrid::writeStructuredFields: "
          << "No entry in fieldDims for field '" << field.name()
          << "' with " << nLevField << " levels.";
      ABORT(oss.str());
    }
    const auto &dims = it->second;

    // Look for fieldname in the iofile configuration and use the value if key found
    const std::string fieldLong = field.name();
    const char * fieldLongC = fieldLong.c_str();

    // Get the fieldmetadata for this field
    const FieldMetadata & fieldMetadata = geom_.fieldsMetaData().getFieldMetadata(fieldLong);
    std::string unitsStr = fieldMetadata.getVarUnits();
    const char * units = unitsStr.c_str();

    std::string fieldName = fieldLong;
    if (ioNames.has(fieldName)) {
      fieldName = ioNames.getString(fieldLong);
    }

    // Define the field in the file
    nc_rc(nc_def_var(fileId, fieldName.c_str(), ncPrec, dims.size(), dims.data(), &fIv),
          "nc_def_var " + fieldName);
    nc_rc(nc_put_att_text(fileId, fIv, "units", strlen(units), units),
          "nc_put_att_text " + fieldName + " units");
    nc_rc(nc_put_att_text(fileId, fIv, "long_name", strlen(fieldLongC), fieldLongC),
          "nc_put_att_text " + fieldName + " long_name");

    // Insert field into the fieldIvs map
    fieldIvs[field.name()] = fIv;
  }

  // Write ak/bk to the file as global attributes
  // --------------------------------------------
  nc_rc(nc_put_att_double(fileId, NC_GLOBAL, "ak", NC_DOUBLE, ak.size(), ak.data()),
          "nc_put_att_double (ak)");
  nc_rc(nc_put_att_double(fileId, NC_GLOBAL, "bk", NC_DOUBLE, bk.size(), bk.data()),
          "nc_put_att_double (bk)");
  nc_rc(nc_put_att_text(fileId, NC_GLOBAL, "grid", strlen(gridStr_.c_str()), gridStr_.c_str()),
          "nc_put_att_text (grid)");
  nc_rc(nc_put_att_int(fileId, NC_GLOBAL, "im", NC_INT, 1, &nLon),
          "nc_put_att_int (im)");
  nc_rc(nc_put_att_int(fileId, NC_GLOBAL, "jm", NC_INT, 1, &nLat),
          "nc_put_att_int (im)");

  // End definition mode
  // -------------------
  nc_rc(nc_enddef(fileId), "nc_enddef");

  // Write coordinate data into the file
  // -----------------------------------
  nc_rc(nc_put_var_double(fileId, fieldIvs[params_.latName.value()], latArr.data()),
        "nc_put_var_double (lat)");
  nc_rc(nc_put_var_double(fileId, fieldIvs[params_.lonName.value()], lonArr.data()),
        "nc_put_var_double (lon)");
  nc_rc(nc_put_var_int(fileId, fieldIvs[params_.levName.value()], levArr.data()),
        "nc_put_var_int (lev)");
  nc_rc(nc_put_var_int(fileId, fieldIvs[params_.edgName.value()], edgArr.data()),
        "nc_put_var_int (edg)");
  nc_rc(nc_put_var_int(fileId, fieldIvs[params_.timName.value()], timArr.data()),
        "nc_put_var_int (tim)");

  // Write the fields into the file
  // ------------------------------
  for (auto& field : fields) {
    // Get number of levels for this field
    const int nLevField = field.shape(1);

    // Create a rank 2 view of the field
    const auto fieldView = atlas::array::make_view<double, 2>(field);

    // Vector to hold the packed field
    std::vector<double> values(nLat*nLon*nLevField);

    // Loop over dimensions and pack the field
    for (size_t k = 0; k < nLevField; ++k) {
      for (size_t j = 0; j < nLat; ++j) {
        for (size_t i = 0; i < nLon; ++i) {
          values[k*nLat*nLon + j*nLon + i] = fieldView((nLat - 1 - j) * nLon + i, k);
        }
      }
    }

    // Write the field to the file
    nc_rc(nc_put_var_double(fileId, fieldIvs[field.name()], values.data()),
          "nc_put_var_double " + field.name());
  }

  // Close netCDF file
  // -----------------
  nc_rc(nc_close(fileId), "nc_close");
}

// -------------------------------------------------------------------------------------------------

/// Reads one 2D or 3D variable from an open NetCDF file into a newly created Atlas Field.
///
/// Supported dimension orderings:
///   4-D: (time, pfull, grid_yt, grid_xt)  → 3D field, reads slab at time=0
///   3-D: (time, grid_yt, grid_xt)         → 2D surface field, reads slab at time=0
///
/// The variable is read as float and converted to double.  Missing values are detected in
/// two complementary ways:
///   1. Absolute threshold: any |val| > 0.5 × kGufsFillValue (= 9.99e20) is flagged.
///   2. Exact match: any val equal to the _FillValue or missing_value attribute is flagged.
/// Either condition triggers an ABORT (abort-on-missing policy).
///
/// @param fileId       Open read-mode NetCDF file ID.
/// @param varName      Name of the variable to read from the file.
/// @param nLat         Expected number of latitude rows  (jm).
/// @param nLon         Expected number of longitude columns (im).
/// @param latDimName   Name of the latitude  dimension in the file (e.g. "grid_yt").
/// @param lonDimName   Name of the longitude dimension in the file (e.g. "grid_xt").
/// @param flipJ        When true the file j-index is reversed before mapping to the Atlas field:
///                     j_atlas = (nLat - 1 - j_file).  Set by the caller after detecting
///                     whether the file latitude axis runs south-to-north.
/// @return             New Atlas field with shape (npts=nLat*nLon, nlev) on readFunctionSpace_.
atlas::Field IOStructuredGrid::readVarToStructuredAtlasField(
    int fileId, const std::string & varName,
    int nLat, int nLon,
    const std::string & latDimName,
    const std::string & lonDimName,
    bool flipJ) const {
  // GFS/UFS NetCDF files use 9.99e20 as the _FillValue / missing_value convention.
  // Any value whose |val| exceeds half that magnitude is treated as missing.
  static constexpr float kGufsFillValue  = 9.99e20f;
  static constexpr float kMissingThreshold = 0.5f * kGufsFillValue;

  // Step 1: Get variable ID
  // -----------------------
  int varId;
  nc_rc(nc_inq_varid(fileId, varName.c_str(), &varId), "nc_inq_varid " + varName);

  // Step 2: Get number of dimensions
  // ---------------------------------
  int ndims;
  nc_rc(nc_inq_varndims(fileId, varId, &ndims), "nc_inq_varndims " + varName);
  if (ndims < 2 || ndims > 4) {
    std::ostringstream oss;
    oss << "IOStructuredGrid::readVarToStructuredAtlasField: variable '" << varName
        << "' has " << ndims << " dims; expected 2, 3, or 4.";
    ABORT(oss.str());
  }

  // Step 3: Get dimension IDs, names, and lengths
  // ----------------------------------------------
  std::vector<int> dimids(ndims);
  nc_rc(nc_inq_vardimid(fileId, varId, dimids.data()), "nc_inq_vardimid " + varName);

  std::vector<std::string> dimNames(ndims);
  std::vector<size_t> dimLens(ndims);
  for (int d = 0; d < ndims; ++d) {
    char dname[NC_MAX_NAME + 1];
    nc_rc(nc_inq_dimname(fileId, dimids[d], dname), "nc_inq_dimname " + varName);
    nc_rc(nc_inq_dimlen(fileId, dimids[d], &dimLens[d]),  "nc_inq_dimlen "  + varName);
    dimNames[d] = std::string(dname);
    oops::Log::trace() << classname() << "  dim[" << d << "] = '" << dimNames[d]
                       << "' len=" << dimLens[d] << std::endl;
  }

  // Step 4: Identify roles of each dimension
  // Supported layouts:
  //   4-D: (time, pfull, grid_yt, grid_xt)  ← iTime=0, iLev=1, iLat=2, iLon=3
  //   3-D: (time, grid_yt, grid_xt)         ← iTime=0, iLat=1, iLon=2
  // Any unrecognised dimension (not time, not lat, not lon) is treated as the vertical.
  // ----------------------------------------------------------------------------------
  int iTime = -1, iLev = -1, iLat = -1, iLon = -1;
  for (int d = 0; d < ndims; ++d) {
    const std::string & dn = dimNames[d];
    if (dn == "time" || dn == "Time" || dn == "TIME") {
      iTime = d;
    } else if (dn == latDimName) {
      iLat = d;
    } else if (dn == lonDimName) {
      iLon = d;
    } else {
      // Treat any other dimension as the vertical level
      iLev = d;
    }
  }

  if (iLat == -1) {
    ABORT("IOStructuredGrid::readVarToStructuredAtlasField: cannot identify lat dimension '"
          + latDimName + "' in variable '" + varName + "'");
  }
  if (iLon == -1) {
    ABORT("IOStructuredGrid::readVarToStructuredAtlasField: cannot identify lon dimension '"
          + lonDimName + "' in variable '" + varName + "'");
  }

  // Validate that lat and lon sizes match the expected grid
  if (static_cast<int>(dimLens[iLat]) != nLat) {
    std::ostringstream oss;
    oss << "IOStructuredGrid::readVarToStructuredAtlasField: variable '" << varName
        << "' lat dim '" << dimNames[iLat] << "' size " << dimLens[iLat]
        << " != expected nLat=" << nLat;
    ABORT(oss.str());
  }
  if (static_cast<int>(dimLens[iLon]) != nLon) {
    std::ostringstream oss;
    oss << "IOStructuredGrid::readVarToStructuredAtlasField: variable '" << varName
        << "' lon dim '" << dimNames[iLon] << "' size " << dimLens[iLon]
        << " != expected nLon=" << nLon;
    ABORT(oss.str());
  }

  const int nLevField = (iLev >= 0) ? static_cast<int>(dimLens[iLev]) : 1;
  oops::Log::trace() << classname() << " readVarToStructuredAtlasField: var='" << varName
                     << "' nLat=" << nLat << " nLon=" << nLon
                     << " nLev=" << nLevField << " flipJ=" << flipJ << std::endl;

  // Step 5: Read _FillValue / missing_value attributes (both optional).
  // If the attribute is not present, default to the standard GFS/UFS fill value (9.99e20).
  // Values matching either attribute (within a relative tolerance) or exceeding the
  // kMissingThreshold are treated as missing and trigger an ABORT.
  // -------------------------------------------------------------------------
  float fillValue   = kGufsFillValue;
  float missingValue = kGufsFillValue;
  {
    float tmp;
    int rc = nc_get_att_float(fileId, varId, "_FillValue", &tmp);
    if (rc == NC_NOERR) fillValue = tmp;
    else if (rc != NC_ENOTATT)
      nc_rc(rc, "nc_get_att_float _FillValue " + varName);

    rc = nc_get_att_float(fileId, varId, "missing_value", &tmp);
    if (rc == NC_NOERR) missingValue = tmp;
    else if (rc != NC_ENOTATT)
      nc_rc(rc, "nc_get_att_float missing_value " + varName);
  }

  // Step 6: Determine the local j-row range owned by this MPI rank.
  // readFunctionSpace_ uses a latitude-band distribution, so each rank owns a contiguous
  // band [j_begin, j_end) of Atlas latitude rows with ALL lon columns.  Atlas j=0 is the northernmost row.
  // flipJ mapping:
  //   flipJ=false → file j = Atlas j  → read file rows [j_begin, j_end)
  //   flipJ=true  → file j = nLat-1-j_atlas → read file rows [nLat-j_end, nLat-j_begin)
  // -------------------------------------------------------------------------
  const int j_begin = static_cast<int>(readFunctionSpace_->j_begin());
  const int j_end   = static_cast<int>(readFunctionSpace_->j_end());
  const int n_j_local = j_end - j_begin;

  // Build start/count arrays for reading only the local j-rows from the file.
  std::vector<size_t> start(ndims, 0);
  std::vector<size_t> count(ndims, 1);
  if (iTime >= 0) { start[iTime] = 0; count[iTime] = 1; }
  if (iLev  >= 0) { start[iLev]  = 0; count[iLev]  = static_cast<size_t>(nLevField); }
  start[iLon] = 0; count[iLon] = static_cast<size_t>(nLon);

  // File j-row range corresponding to this rank's Atlas rows
  const int j_file_beg = flipJ ? (nLat - j_end) : j_begin;
  start[iLat] = static_cast<size_t>(j_file_beg);
  count[iLat] = static_cast<size_t>(n_j_local);

  const size_t bufSize = static_cast<size_t>(nLevField) * n_j_local * nLon;
  std::vector<float> buf(bufSize);
  if (n_j_local > 0) {
    nc_rc(nc_get_vara_float(fileId, varId, start.data(), count.data(), buf.data()),
          "nc_get_vara_float " + varName);
  }

  // Step 7: Create the Atlas field on the local portion of readFunctionSpace_.
  // The field has shape (local_npts, nLevField) where local_npts = readFunctionSpace_->size().
  // ----------------------------------------------------------------------
  atlas::Field field = readFunctionSpace_->createField<double>(
      atlas::option::name(varName) | atlas::option::levels(nLevField));
  auto fieldView = atlas::array::make_view<double, 2>(field);

  // Step 8: Compute buffer strides (C-order, based on the local count array).
  // Using strides makes the indexing correct regardless of the dimension ordering in the file.
  // ----------------------------------------------------------------------------------------
  std::vector<size_t> strides(ndims, 1);
  for (int d = ndims - 2; d >= 0; --d) {
    strides[d] = strides[d + 1] * count[d + 1];
  }

  // Step 9: Check missing values, convert float → double, pack into Atlas field view.
  // Atlas StructuredColumns::index(i, j) has signature (lon_col_index, lat_row_index).
  // Note: i = longitude column (0-based), j = latitude row (0-based) — column FIRST.
  // The local flat index for owned point (i, j_atlas) is accessed as fieldView(idx, k)
  // where idx = readFunctionSpace_->index(i, j_atlas).
  //
  // Buffer row index (buffer_row) within the read buffer for Atlas row j_atlas:
  //   flipJ=false: buffer_row = j_atlas - j_begin  (buffer rows in increasing Atlas-j order)
  //   flipJ=true:  buffer_row = j_end - 1 - j_atlas  (buffer rows in decreasing Atlas-j order)
  //                because file rows [nLat-j_end, nLat-j_begin) correspond to Atlas rows
  //                [j_begin, j_end) in reversed order.
  // -----------------------------------------------------------------------------------
  double valMin =  std::numeric_limits<double>::max();
  double valMax = -std::numeric_limits<double>::max();

  for (int k = 0; k < nLevField; ++k) {
    for (int j_atlas = j_begin; j_atlas < j_end; ++j_atlas) {
      const int buffer_row = flipJ ? (j_end - 1 - j_atlas) : (j_atlas - j_begin);
      for (int i = 0; i < nLon; ++i) {
        // Buffer index using precomputed strides; time index is always 0
        const size_t bufIdx = (iLev  >= 0 ? static_cast<size_t>(k) * strides[iLev]  : 0)
                            + static_cast<size_t>(buffer_row) * strides[iLat]
                            + static_cast<size_t>(i) * strides[iLon];
        const float val = buf[bufIdx];
        // Check against the threshold AND against the explicit fill/missing values.
        // The threshold test catches the standard GFS/UFS 9.99e20 fill value.
        // The attribute tests catch non-standard fill values (exact float comparison
        // is intentional here: fill values are written and read as exact bit patterns).
        const bool isMissing = (std::abs(val) > kMissingThreshold)
                             || (val == fillValue)
                             || (val == missingValue);
        if (isMissing) {
          std::ostringstream oss;
          oss << "IOStructuredGrid::readVarToStructuredAtlasField: missing/fill value "
              << val << " detected in variable '" << varName
              << "' at (j_atlas=" << j_atlas << ", i=" << i << ", k=" << k << "). "
              << "Policy: abort. Check the input file for corrupted or unfilled data.";
          ABORT(oss.str());
        }
        const double dval = static_cast<double>(val);
        // Atlas StructuredColumns::index(i, j): i=longitude column, j=latitude row.
        // Correct call: index(i, j_atlas) not index(j_atlas, i).
        const atlas::idx_t atlasIdx = readFunctionSpace_->index(i, j_atlas);
        // Guard: detect any out-of-range Atlas index early (safety net in case the
        // readFunctionSpace_ grid still doesn't match the file, or if j_atlas is outside
        // the owned [j_begin, j_end) range — which would indicate a logic error above).
        if (atlasIdx < 0 || atlasIdx >= static_cast<atlas::idx_t>(fieldView.shape(0))) {
          std::ostringstream oss;
          oss << "IOStructuredGrid::readVarToStructuredAtlasField: Atlas index "
              << atlasIdx << " out of range [0, " << fieldView.shape(0) << ") "
              << "for variable '" << varName << "' at "
              << "(j_atlas=" << j_atlas << ", i=" << i << ", k=" << k << "). "
              << "File grid: nLat=" << nLat << " nLon=" << nLon
              << ", readFunctionSpace_ grid='" << readFunctionSpace_->grid().name()
              << "', j_begin=" << j_begin << " j_end=" << j_end << ".";
          ABORT(oss.str());
        }
        fieldView(atlasIdx, k) = dval;
        if (dval < valMin) valMin = dval;
        if (dval > valMax) valMax = dval;
      }
    }
  }

  oops::Log::trace() << classname() << "  '" << varName << "' range: ["
                     << valMin << ", " << valMax << "]" << std::endl;
  return field;
}

// -------------------------------------------------------------------------------------------------

/// Opens input file(s) and reads the requested variables into newly created Atlas fields
/// that are added to outFields.  Called on ALL MPI ranks.
///
/// readFunctionSpace_ uses a latitude-band distribution: each MPI rank owns a contiguous
/// band of complete latitude rows [j_begin, j_end) and reads only those rows from the NC file.
/// This ensures GeometryData builds its globalNodeTree_ on all ranks, which is required
/// by GlobalInterpolator::apply() (closestTask() asserts !globalNodeTree_.empty()).
/// After all ranks have filled their local portion, readInterpolator_->apply() performs a
/// distributed StructuredColumns → CubeSphere interpolation across all ranks.
///
/// File selection follows the same policy as write:
///   - If params_.filenames is non-empty, each entry (prefixed by params_.datapath) is opened
///     in sequence; duplicate variable names across files are resolved with a first-file-wins
///     policy and a trace log is emitted for each duplicate field that is skipped.
///   - Otherwise, params_.filename (after datetime formatting) is used.
///
/// @param[out] outFields   Atlas FieldSet; read fields are appended to it.
/// @param fieldNames       Long names of the fields to read (from State/Increment variables).
/// @param time             Valid time used to format the filename template.
/// @param ioNames          Configuration mapping long field names to in-file variable names.
/// @param fileTime         If non-null, receives the valid time read from the first file
///                         (time_iso preferred; falls back to time + units).
void IOStructuredGrid::readStructuredFields(
    atlas::FieldSet & outFields,
    const std::vector<std::string> & fieldNames,
    const util::DateTime & time,
    const eckit::LocalConfiguration & ioNames,
    util::DateTime * fileTime) const {
  // Build the ordered list of input file paths
  // ------------------------------------------
  std::vector<std::string> inputFiles;
  const std::vector<std::string> filenamesList = params_.filenames.value();
  const std::string datapathStr = params_.datapath.value();

  if (!filenamesList.empty()) {
    for (const auto & fn : filenamesList) {
      inputFiles.push_back(datapathStr.empty() ? fn : datapathStr + "/" + fn);
    }
  } else {
    std::string pathFile = params_.filename.value();
    if (pathFile.find("%Y") == std::string::npos) pathFile += "%Y%m%d_%H%M%Sz";
    if (pathFile.find(".nc")  == std::string::npos) pathFile += ".nc4";
    pathFile = time.formatString(pathFile);
    util::stringfunctions::swapNameMember(params_.toConfiguration(), pathFile);
    if (!datapathStr.empty()) pathFile = datapathStr + "/" + pathFile;
    inputFiles.push_back(pathFile);
  }

  const std::string latDimName = params_.readLatName.value();
  const std::string lonDimName = params_.readLonName.value();

  // Track which fields have been successfully read (first-file-wins for duplicates)
  std::set<std::string> fieldsRead;

  // Process each file
  // -----------------
  bool firstFile = true;
  // Grid string determined from the first file; subsequent files must match.
  // Mixing files with different grids in one read call is not supported because
  // the resulting Atlas fields would live on different function spaces and could
  // not be interpolated together by a single readInterpolator_.
  std::string expectedGridStr;
  for (const auto & pathFile : inputFiles) {
    oops::Log::trace() << classname() << " readStructuredFields: opening " << pathFile
                       << std::endl;

    int fileId;
    nc_rc(nc_open(pathFile.c_str(), NC_NOWRITE, &fileId), "nc_open " + pathFile);

    // Read the valid time from the first file when requested by the caller.
    if (firstFile && fileTime != nullptr) {
      readValidTimeFromFile(fileId, *fileTime);
      firstFile = false;
    }

    // Read grid dimensions — prefer global attributes im/jm, fall back to dim lengths
    const int nLon = readGlobalIntAttrOrDimLen(fileId, "im", lonDimName);
    const int nLat = readGlobalIntAttrOrDimLen(fileId, "jm", latDimName);
    oops::Log::trace() << classname() << "  nLat=" << nLat << " nLon=" << nLon << std::endl;

    // -----------------------------------------------------------------------
    // Ensure readFunctionSpace_ (and readInterpolator_) match this file's grid.
    //
    // The readFunctionSpace_ is initialised in the constructor from the FV3
    // geometry (F<geom.npy()-1>).  Input files may come from a different-resolution
    // Gaussian grid run (e.g. a higher-resolution GFS background for a coarser
    // analysis), in which case the per-row longitude count (nx) differs between
    // the Atlas function space and the file.  When nLon_file > atlas_nx the call
    //   readFunctionSpace_->index(j_atlas, i)  for i >= atlas_nx
    // returns an index that exceeds the Atlas field size, writing past the end of
    // the field buffer (array overflow → segmentation fault after many iterations).
    //
    // Fix: infer the Atlas Gaussian grid that matches the file's nLon/nLat and
    // lazily rebuild readFunctionSpace_ and readInterpolator_ when needed.
    // The function space and interpolator are marked mutable for this purpose.
    // -----------------------------------------------------------------------
    if (gridStr_ == "gaussian") {
      const std::string fileGridStr = inferAtlasGaussianGridString(nLon, nLat);
      // Enforce that all input files in this call use the same Gaussian grid.
      // Fields from different grids would live on different Atlas function spaces
      // and cannot be interpolated together by a single readInterpolator_.
      if (expectedGridStr.empty()) {
        expectedGridStr = fileGridStr;
      } else if (fileGridStr != expectedGridStr) {
        std::ostringstream oss;
        oss << "IOStructuredGrid::readStructuredFields: input file '" << pathFile
            << "' is on grid '" << fileGridStr
            << "' but an earlier file was on grid '" << expectedGridStr
            << "'. All input files in a single read call must use the same Gaussian grid.";
        ABORT(oss.str());
      }
      // readFunctionSpace_ is always initialised by the constructor, so accessing
      // its grid name here is always safe.
      if (fileGridStr != readFunctionSpace_->grid().name()) {
        const std::string prevGridStr = readFunctionSpace_->grid().name();
        oops::Log::info() << classname()
                          << " readStructuredFields: input file grid '" << fileGridStr
                          << "' differs from current readFunctionSpace_ grid '" << prevGridStr
                          << "'; rebuilding readFunctionSpace_ and readInterpolator_."
                          << std::endl;
        const atlas::Grid fileGrid(fileGridStr);
        eckit::LocalConfiguration atlas_conf;
        atlas_conf.set("mpi_comm", geom_.getComm().name());
        // Use the same 1-D latitude-band distribution as the constructor to ensure:
        //   (a) all ranks own complete rows (i_begin(j)=0, i_end(j)=nx(j)) so the NC
        //       reading loop over i in [0, nLon) stays within each rank's owned range.
        //   (b) every rank owns > 0 points so GeometryData builds its globalNodeTree_.
        const atlas::grid::Distribution fileReadDist =
            makeLatBandDistribution(fileGrid, geom_.getComm().size());
        readFunctionSpace_.reset(new atlas::functionspace::StructuredColumns(
            fileGrid, fileReadDist, atlas_conf));
        // An empty FieldSet is sufficient: StructuredColumns provides its own
        // coordinate information so GeometryData does not need extra fields.
        atlas::FieldSet readGeomFields;
        oops::GeometryData readGeomData(*readFunctionSpace_, readGeomFields,
                                        geom_.levelsAreTopDown(), geom_.getComm());
        readInterpolator_.reset(new oops::GlobalInterpolator(
            params_.toConfiguration(), readGeomData,
            geom_.functionSpace(), geom_.getComm()));
        oops::Log::info() << classname()
                          << " readStructuredFields: readFunctionSpace_ rebuilt for grid '"
                          << fileGridStr << "'." << std::endl;
      }
    } else {
      // For non-Gaussian grids validate that the file dimensions match the
      // function space that was created at construction time.
      const atlas::StructuredGrid fsGrid(readFunctionSpace_->grid());
      const int fsNy = static_cast<int>(fsGrid.ny());
      const int fsNx = static_cast<int>(fsGrid.nx(0));
      if (nLat != fsNy || nLon != fsNx) {
        std::ostringstream oss;
        oss << "IOStructuredGrid::readStructuredFields: input file '" << pathFile
            << "' has nLat=" << nLat << " nLon=" << nLon
            << " but readFunctionSpace_ was built for grid '"
            << readFunctionSpace_->grid().name()
            << "' with ny=" << fsNy << " nx=" << fsNx
            << ". Ensure 'gridtype' in the YAML matches the input file's grid.";
        ABORT(oss.str());
      }
    }

    // Detect latitude orientation: north-first (N→S) or south-first (S→N).
    // Atlas StructuredColumns j=0 is always the northernmost row (N→S ordering).
    // flipJ is set to true when the file stores latitudes south-to-north so that
    // j_file=0 (south) is remapped to j_atlas=nLat-1 (south in Atlas).
    const bool fileNorthFirst = detectFileLatNorthFirst(fileId, latDimName, nLat);
    const bool flipJ = !fileNorthFirst;
    oops::Log::trace() << classname() << "  fileNorthFirst=" << fileNorthFirst
                       << " flipJ=" << flipJ << std::endl;

    for (const auto & fieldLong : fieldNames) {
      // Skip fields already filled from an earlier file (first-file-wins policy ensures
      // deterministic behaviour when the same variable appears in more than one input file).
      if (fieldsRead.count(fieldLong)) {
        oops::Log::trace() << classname() << "  field '" << fieldLong
                           << "' already read from an earlier file; skipping in "
                           << pathFile << std::endl;
        continue;
      }

      // Resolve the in-file variable name (may differ from the long name)
      std::string varName = fieldLong;
      if (ioNames.has(fieldLong)) varName = ioNames.getString(fieldLong);

      // Check whether the variable exists in this file
      int varId;
      const int rc = nc_inq_varid(fileId, varName.c_str(), &varId);
      if (rc == NC_ENOTVAR) {
        oops::Log::trace() << classname() << "  variable '" << varName
                           << "' not found in " << pathFile << std::endl;
        continue;
      }
      nc_rc(rc, "nc_inq_varid " + varName);

      oops::Log::trace() << classname() << "  reading '" << varName
                         << "' for field '" << fieldLong << "'" << std::endl;

      atlas::Field field = this->readVarToStructuredAtlasField(
          fileId, varName, nLat, nLon, latDimName, lonDimName, flipJ);

      // Give the field the long name so it can be matched back to the State/Increment
      field.rename(fieldLong);
      outFields.add(field);
      fieldsRead.insert(fieldLong);

      // Log the successful sourcing so operators can trace which file each field came from
      oops::Log::info() << classname() << " field '" << fieldLong
                        << "' (file var '" << varName << "') sourced from " << pathFile
                        << std::endl;
    }

    nc_rc(nc_close(fileId), "nc_close " + pathFile);
  }

  // Warn about any fields that were not found in any input file
  for (const auto & fieldLong : fieldNames) {
    if (!fieldsRead.count(fieldLong)) {
      oops::Log::warning() << classname() << " readStructuredFields: field '"
                           << fieldLong << "' was not found in any input file" << std::endl;
    }
  }
}

// -------------------------------------------------------------------------------------------------

void IOStructuredGrid::print(std::ostream & os) const {
  os << classname() << " IO using Atlas Structured Grid";
}

// -------------------------------------------------------------------------------------------------

}  // namespace fv3jedi
