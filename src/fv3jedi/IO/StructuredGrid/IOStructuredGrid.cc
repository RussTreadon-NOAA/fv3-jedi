/*
 * (C) Copyright 2025- UCAR.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#include <netcdf.h>

#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <vector>

#include "atlas/functionspace.h"
#include "atlas/grid.h"

#include "oops/base/GeometryData.h"
#include "oops/base/Variables.h"
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
IOStructuredGrid::IOStructuredGrid(const Geometry & geom, const Parameters_ & params)
  : IOBase(geom, params.toConfiguration()), interpolator_(), params_(params), gridStr_(""),
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
  // The read function space uses the same serial distribution (all grid points on rank 0),
  // mirroring the write function space. It is built once here and cached for all reads.
  // It is kept as a separate member from writeFunctionSpace_ so that future steps can give it
  // a different distribution (e.g. equal_regions for parallel reads) without affecting writes.
  // --------------------------------------------------------------------------------------
  readFunctionSpace_.reset(new atlas::functionspace::StructuredColumns(grid, dist, atlas_conf));

  // Create a GeometryData object
  // ----------------------------
  oops::GeometryData geomData(geom.functionSpace(), geom.fields(), geom.levelsAreTopDown(),
                              geom.getComm());

  // Create a generic interpolator for converting to the structured grid
  // -------------------------------------------------------------------
  interpolator_.reset(new oops::GlobalInterpolator(params.toConfiguration(), geomData,
                                                   *writeFunctionSpace_,
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
  const std::vector<std::string> fieldNames(vars.variables().begin(), vars.variables().end());

  // Read fields from file(s) into the geographic (readFunctionSpace_) Atlas FieldSet on rank 0.
  // The readFunctionSpace_ has a serial distribution (all points on rank 0), so reading is
  // done on rank 0 only.
  atlas::FieldSet fieldsGeographic;
  if (geom_.getComm().rank() == 0) {
    this->readStructuredFields(fieldsGeographic, fieldNames, x.validTime(), fileionames);
  }

  // Reverse interpolation (geographic → cube sphere) and State::fromFieldSet are not yet
  // implemented; they will be added in a subsequent step.
  ABORT("IOStructuredGrid::read(State): reverse interpolation not yet implemented");
}

// -------------------------------------------------------------------------------------------------

void IOStructuredGrid::read(Increment & dx, const eckit::LocalConfiguration & fileionames,
                                const eckit::LocalConfiguration & fileioscaling) const {
  util::Timer timer(classname(), "read increment");
  oops::Log::trace() << classname() << " read increment starting" << std::endl;

  // Collect the field names requested by the Increment
  const oops::Variables & vars = dx.variables();
  const std::vector<std::string> fieldNames(vars.variables().begin(), vars.variables().end());

  // Read fields from file(s) into the geographic (readFunctionSpace_) Atlas FieldSet on rank 0.
  atlas::FieldSet fieldsGeographic;
  if (geom_.getComm().rank() == 0) {
    this->readStructuredFields(fieldsGeographic, fieldNames, dx.validTime(), fileionames);
  }

  // Reverse interpolation (geographic → cube sphere) and Increment::fromFieldSet are not yet
  // implemented; they will be added in a subsequent step.
  ABORT("IOStructuredGrid::read(Increment): reverse interpolation not yet implemented");
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

  // Step 6: Build start/count arrays and read variable data as float
  // ----------------------------------------------------------------
  std::vector<size_t> start(ndims, 0);
  std::vector<size_t> count(ndims, 1);
  if (iTime >= 0) { start[iTime] = 0; count[iTime] = 1; }
  if (iLev  >= 0) { start[iLev]  = 0; count[iLev]  = static_cast<size_t>(nLevField); }
  start[iLat] = 0; count[iLat] = static_cast<size_t>(nLat);
  start[iLon] = 0; count[iLon] = static_cast<size_t>(nLon);

  const size_t bufSize = static_cast<size_t>(nLevField) * nLat * nLon;
  std::vector<float> buf(bufSize);
  nc_rc(nc_get_vara_float(fileId, varId, start.data(), count.data(), buf.data()),
        "nc_get_vara_float " + varName);

  // Step 7: Create the Atlas field with shape (npts=nLat*nLon, nLevField)
  // ----------------------------------------------------------------------
  atlas::Field field = readFunctionSpace_->createField<double>(
      atlas::option::name(varName) | atlas::option::levels(nLevField));
  auto fieldView = atlas::array::make_view<double, 2>(field);

  // Step 8: Compute buffer strides (C-order, dimension ordering as in file).
  // Using strides makes the indexing correct regardless of the dimension
  // ordering in the file.
  // ------------------------------------------------------------------------
  std::vector<size_t> strides(ndims, 1);
  for (int d = ndims - 2; d >= 0; --d) {
    strides[d] = strides[d + 1] * count[d + 1];
  }

  // Step 9: Check missing values, convert float → double, pack into Atlas field view.
  // Atlas field layout: fieldView(j*nLon + i, k) = value at (lat j, lon i, level k).
  // When flipJ is true the file stores rows south-to-north but Atlas expects north-to-south
  // (j=0 = northernmost row), so the file row j_file maps to Atlas row (nLat-1-j_file).
  // -----------------------------------------------------------------------------------
  double valMin =  std::numeric_limits<double>::max();
  double valMax = -std::numeric_limits<double>::max();

  for (int k = 0; k < nLevField; ++k) {
    for (int j_file = 0; j_file < nLat; ++j_file) {
      const int j_atlas = flipJ ? (nLat - 1 - j_file) : j_file;
      for (int i = 0; i < nLon; ++i) {
        // Buffer index using precomputed strides; time index is always 0
        size_t bufIdx = (iLev  >= 0 ? static_cast<size_t>(k) * strides[iLev]  : 0)
                      + static_cast<size_t>(j_file) * strides[iLat]
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
              << "' at (j=" << j_file << ", i=" << i << ", k=" << k << "). "
              << "Policy: abort. Check the input file for corrupted or unfilled data.";
          ABORT(oss.str());
        }
        const double dval = static_cast<double>(val);
        fieldView(j_atlas * nLon + i, k) = dval;
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
/// that are added to outFields.  Should only be called on rank 0 because readFunctionSpace_
/// uses a serial distribution (all grid points on rank 0).
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
void IOStructuredGrid::readStructuredFields(
    atlas::FieldSet & outFields,
    const std::vector<std::string> & fieldNames,
    const util::DateTime & time,
    const eckit::LocalConfiguration & ioNames) const {
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
  for (const auto & pathFile : inputFiles) {
    oops::Log::trace() << classname() << " readStructuredFields: opening " << pathFile
                       << std::endl;

    int fileId;
    nc_rc(nc_open(pathFile.c_str(), NC_NOWRITE, &fileId), "nc_open " + pathFile);

    // Read grid dimensions — prefer global attributes im/jm, fall back to dim lengths
    const int nLon = readGlobalIntAttrOrDimLen(fileId, "im", lonDimName);
    const int nLat = readGlobalIntAttrOrDimLen(fileId, "jm", latDimName);
    oops::Log::trace() << classname() << "  nLat=" << nLat << " nLon=" << nLon << std::endl;

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
