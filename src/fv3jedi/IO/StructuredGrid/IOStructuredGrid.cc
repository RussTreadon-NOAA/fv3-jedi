/*
 * (C) Copyright 2025- UCAR.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#include <netcdf.h>

#include <map>
#include <set>
#include <vector>

#include "atlas/functionspace.h"
#include "atlas/grid.h"

#include "oops/base/GeometryData.h"
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
    geom_(geom), writeFunctionSpace_() {
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

  // Structured grid function space
  // ------------------------------
  writeFunctionSpace_.reset(new atlas::functionspace::StructuredColumns(grid, dist, atlas_conf));

  // Create a GeometryData object
  // ----------------------------
  oops::GeometryData geomData(geom.functionSpace(), geom.fields(), geom.levelsAreTopDown(),
                              geom.getComm());

  // Create a generic interpolator for converting to the structured grid
  // -------------------------------------------------------------------
  interpolator_.reset(new oops::GlobalInterpolator(params.toConfiguration(), geomData,
                                                   *writeFunctionSpace_,
                                                   geom.getComm()));

  // Create a balanced StructuredColumns function space for reading
  // (balanced distribution so every rank owns points, required for GeometryData triangulation)
  // ------------------------------------------------------------------------------------------
  readFunctionSpace_.reset(new atlas::functionspace::StructuredColumns(grid, atlas_conf));

  // Build GeometryData from the balanced StructuredColumns (source for read interpolation)
  // ---------------------------------------------------------------------------------------
  oops::GeometryData readGeomData(*readFunctionSpace_, atlas::FieldSet(),
                                  geom.levelsAreTopDown(), geom.getComm());

  // Create interpolator for reading: balanced structured grid → cube sphere NodeColumns
  // ------------------------------------------------------------------------------------
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
  this->readAndInterp(x, "state", fileionames, fileioscaling);
  oops::Log::trace() << classname() << " read state done" << std::endl;
  }

// -------------------------------------------------------------------------------------------------

void IOStructuredGrid::read(Increment & dx, const eckit::LocalConfiguration & fileionames,
                                const eckit::LocalConfiguration & fileioscaling) const {
  util::Timer timer(classname(), "read increment");
  oops::Log::trace() << classname() << " read increment starting" << std::endl;
  this->readAndInterp(dx, "increment", fileionames, fileioscaling);
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

template <typename T>
void IOStructuredGrid::readAndInterp(T & obj, const std::string & label,
                                     const eckit::LocalConfiguration & fileionames,
                                     const eckit::LocalConfiguration & fileioscaling) const {
  util::Timer timer(classname(), "read " + label);
  oops::Log::trace() << classname() << " read " << label << " starting" << std::endl;

  // Get the cube sphere fields (determines names and level counts to request)
  atlas::FieldSet fieldsCubeSphere;
  obj.toFieldSet(fieldsCubeSphere);

  // Create corresponding fields on the balanced structured grid
  atlas::FieldSet fieldsGeographic;
  for (const auto & field : fieldsCubeSphere) {
    atlas::Field geoField = readFunctionSpace_->createField(
        atlas::option::name(field.name()) | atlas::option::levels(field.shape(1)) |
        atlas::option::datatype(field.datatype()));
    fieldsGeographic.add(geoField);
  }

  // Each rank reads its own latitude rows from the NetCDF file
  this->readStructuredFields(obj.validTime(), fieldsGeographic, fileionames);

  // Interpolate from the balanced structured grid to the cube sphere
  readInterpolator_->apply(fieldsGeographic, fieldsCubeSphere);

  // Update the State/Increment from the interpolated cube sphere fields
  obj.fromFieldSet(fieldsCubeSphere);

  oops::Log::trace() << classname() << " read " << label << " done" << std::endl;
}

// -------------------------------------------------------------------------------------------------

void IOStructuredGrid::readStructuredFields(const util::DateTime & time,
                                            atlas::FieldSet & fields,
                                            const eckit::LocalConfiguration & ioNames) const {
  util::Timer timer(classname(), "readStructuredFields");
  oops::Log::trace() << classname() << " readStructuredFields starting" << std::endl;

  // Build the ordered list of files to search for fields.
  // Priority: user-specified filenames list (+ datapath) > legacy single-filename fallback.
  std::vector<std::string> pathFiles;
  const std::vector<std::string> & fnList = params_.filenames.value();
  if (!fnList.empty()) {
    const std::string & dp = params_.datapath.value();
    for (const auto & fn : fnList) {
      pathFiles.push_back(dp.empty() ? fn : dp + "/" + fn);
    }
  } else {
    // Fall back to single-file logic compatible with writeStructuredFields output
    std::string pathFile = params_.filename.value();
    if (pathFile.find("%Y") == std::string::npos) {
      pathFile += "%Y%m%d_%H%M%Sz";
    }
    if (pathFile.find(".nc") == std::string::npos) {
      pathFile += ".nc4";
    }
    pathFile = time.formatString(pathFile);
    util::stringfunctions::swapNameMember(params_.toConfiguration(), pathFile);
    pathFiles.push_back(pathFile);
  }

  // Track which fields have been found in at least one file (first-file-wins: once nc_inq_varid
  // succeeds for a field, that field is never searched in any subsequent file, even if the
  // dimensions don't match and the read is skipped for that file).
  std::set<std::string> fieldsFound;
  // Track which fields were actually read (nc_get_vara_float + Atlas fill completed).
  std::set<std::string> fieldsRead;

  // j_beg and j_end depend only on readFunctionSpace_; compute once outside the file loop.
  const int j_beg = readFunctionSpace_->j_begin();
  const int j_end = readFunctionSpace_->j_end();
  const int myNLat = j_end - j_beg;

  oops::Log::trace() << classname() << " j_beg=" << j_beg << " j_end=" << j_end << std::endl;
  oops::Log::trace() << classname() << " myNLat=" << myNLat << std::endl;

  // Expected grid dimensions (set from the first file; subsequent files must match).
  size_t nLatExpected = 0;
  size_t nLonExpected = 0;

  for (const auto & pathFile : pathFiles) {
    // Open this file; all ranks open concurrently (NC_NOWRITE)
    int fileId;
    nc_rc(nc_open(pathFile.c_str(), NC_NOWRITE, &fileId), "nc_open " + pathFile);

    oops::Log::trace() << classname() << " open pathFile=" << pathFile << std::endl;

    // Read grid dimensions from the file
    int latDimId, lonDimId;
    size_t nLat, nLon;
    nc_rc(nc_inq_dimid(fileId, params_.readLatName.value().c_str(), &latDimId),
          "nc_inq_dimid (lat)");
    nc_rc(nc_inq_dimid(fileId, params_.readLonName.value().c_str(), &lonDimId),
          "nc_inq_dimid (lon)");
    nc_rc(nc_inq_dimlen(fileId, latDimId, &nLat), "nc_inq_dimlen (lat)");
    nc_rc(nc_inq_dimlen(fileId, lonDimId, &nLon), "nc_inq_dimlen (lon)");

    oops::Log::trace() << classname() << " nLat=" << nLat << " nLon=" << nLon << std::endl;

    // Validate grid dimensions are consistent across all input files.
    if (nLatExpected == 0) {
      nLatExpected = nLat;
      nLonExpected = nLon;
    } else if (nLat != nLatExpected || nLon != nLonExpected) {
      ABORT("IOStructuredGrid::readStructuredFields: grid dimensions of '" + pathFile +
            "' (" + std::to_string(nLat) + "x" + std::to_string(nLon) + ")"
            + " do not match those of the first input file ("
            + std::to_string(nLatExpected) + "x" + std::to_string(nLonExpected) + ")");
    }

    // Coordinate index convention:
    //
    //   Atlas StructuredColumns j:  j=0 is NORTHERNMOST latitude.
    //   NetCDF file j (nc_j):       nc_j=0 is SOUTHERNMOST latitude (south-to-north storage,
    //                               matching the write convention in writeStructuredFields).
    //   Relationship:  nc_j = nLat - 1 - j_atlas
    //
    // This rank owns Atlas rows [j_beg, j_end).  The matching NetCDF row block is
    //   nc_j in [nLat - j_end,  nLat - 1 - j_beg],  starting at nc_j_start = nLat - j_end.
    //
    // Within the read buffer (size myNLat × nLon per level), local index nc_j_local = 0
    // corresponds to nc_j = nc_j_start (southernmost row of this rank's block), so:
    //   nc_j_local = (nLat - 1 - j_atlas) - (nLat - j_end) = j_end - 1 - j_atlas.
    const size_t nc_j_start = static_cast<size_t>(nLat) - static_cast<size_t>(j_end);

    for (auto & field : fields) {
      // Skip fields already found in an earlier file (first-file-wins).
      if (fieldsFound.count(field.name()) != 0) continue;

      const std::string fieldLong = field.name();
      const int nLevField = field.shape(1);

      // Respect ioNames remapping (symmetrical with writeStructuredFields)
      std::string fieldName = fieldLong;
      if (ioNames.has(fieldLong)) {
        fieldName = ioNames.getString(fieldLong);
      }

      int varId;
      const int rc = nc_inq_varid(fileId, fieldName.c_str(), &varId);
      if (rc != NC_NOERR) {
        // Variable not in this file; try the next file
        continue;
      }

      // Field found in this file — mark it so no subsequent file is searched for it.
      // This enforces first-file-wins semantics even when the ndims check below fails.
      fieldsFound.insert(field.name());

      oops::Log::trace() << classname() << " reading field '" << fieldName
                         << "' nLevField=" << nLevField << std::endl;

      // Validate the variable's dimension count against what the read expects.
      // Surface fields (nLevField==1) expect 3 dims: (time, lat, lon).
      // Multi-level fields (nLevField>1) expect 4 dims: (time, lev, lat, lon).
      int varNdims;
      nc_rc(nc_inq_varndims(fileId, varId, &varNdims), "nc_inq_varndims " + fieldName
            + " in " + pathFile);
      const int expectedNdims = (nLevField == 1) ? 3 : 4;
      if (varNdims != expectedNdims) {
        oops::Log::warning() << classname() << "::readStructuredFields: '"
                             << fieldName << "' has " << varNdims << " dimensions in '"
                             << pathFile << "', expected " << expectedNdims
                             << " -- skipping." << std::endl;
        continue;
      }

      // Read this rank's rows from the file.
      // Variable dims:  surface (nLevField==1): (time, lat, lon)
      //                 multi-level           : (time, lev/edge/four, lat, lon)
      // Use a float buffer: UFS/GFS input files store float32 data.  nc_get_vara_float
      // also handles double-precision NetCDF variables via narrowing conversion (acceptable
      // for NWP initial conditions where float32 precision suffices).
      std::vector<float> values;
      if (nLevField == 1) {
        std::vector<size_t> start = {0, nc_j_start, 0};
        std::vector<size_t> count = {1, static_cast<size_t>(myNLat), nLon};
        values.resize(static_cast<size_t>(myNLat) * nLon);
        nc_rc(nc_get_vara_float(fileId, varId, start.data(), count.data(), values.data()),
              "nc_get_vara_float " + fieldName);
      } else {
        std::vector<size_t> start = {0, 0, nc_j_start, 0};
        std::vector<size_t> count = {1, static_cast<size_t>(nLevField),
                                     static_cast<size_t>(myNLat), nLon};
        values.resize(static_cast<size_t>(nLevField) *
                      static_cast<size_t>(myNLat) * nLon);
        nc_rc(nc_get_vara_float(fileId, varId, start.data(), count.data(), values.data()),
              "nc_get_vara_float " + fieldName);
      }

      oops::Log::trace() << classname() << " read " << values.size() << " floats for '"
                         << fieldName << "' nc_j_start=" << nc_j_start << std::endl;

      oops::Log::trace() << classname() << " filling Atlas view for '" << fieldName
                         << "' datatype=" << field.datatype().str() << std::endl;

      // Fill the Atlas StructuredColumns field view.
      // Buffer: values[k * myNLat * nLon + nc_j_local * nLon + i]
      //   where nc_j_local = j_end - 1 - j_atlas  (see coordinate convention above).
      // Dispatch on the Atlas field's datatype (float32 or float64) using a lambda to
      // avoid duplicating the fill loop.
      const std::string dtypeStr = field.datatype().str();
      auto fillFieldView = [&](auto & fv) {
        using ValueType = std::remove_reference_t<decltype(fv(0, 0))>;
        for (int j = j_beg; j < j_end; ++j) {
          const int nc_j_local = (j_end - 1) - j;
          for (atlas::idx_t i = readFunctionSpace_->i_begin(j);
               i < readFunctionSpace_->i_end(j); ++i) {
            const atlas::idx_t localIdx = readFunctionSpace_->index(j, i);
            for (int k = 0; k < nLevField; ++k) {
              fv(localIdx, k) = static_cast<ValueType>(
                  values[static_cast<size_t>(k) * static_cast<size_t>(myNLat) * nLon
                         + static_cast<size_t>(nc_j_local) * nLon
                         + static_cast<size_t>(i)]);
            }
          }
        }
      };
      if (dtypeStr == "real64") {
        oops::Log::trace() << classname() << " filling real64 view for '" << fieldName
                           << "'" << std::endl;
        auto fieldView = atlas::array::make_view<double, 2>(field);
        fillFieldView(fieldView);
      } else if (dtypeStr == "real32") {
        oops::Log::trace() << classname() << " filling real32 view for '" << fieldName
                           << "'" << std::endl;
        auto fieldView = atlas::array::make_view<float, 2>(field);
        fillFieldView(fieldView);
      } else {
        ABORT("IOStructuredGrid::readStructuredFields: unsupported Atlas field datatype for '"
              + fieldName + "': " + dtypeStr);
      }
      field.set_dirty();
      fieldsRead.insert(field.name());
    }

    nc_rc(nc_close(fileId), "nc_close");
  }

  // Warn for any fields not found in any input file, or found but not read (e.g., ndims mismatch)
  for (const auto & field : fields) {
    if (fieldsRead.count(field.name()) == 0) {
      if (fieldsFound.count(field.name()) != 0) {
        oops::Log::warning() << classname() << "::readStructuredFields: field '"
                             << field.name() << "' was found in an input file but could not be"
                             << " read (dimension mismatch) -- leaving at zero." << std::endl;
      } else {
        oops::Log::warning() << classname() << "::readStructuredFields: field '"
                             << field.name() << "' not found in any input file"
                             << " -- leaving at zero." << std::endl;
      }
    }
  }

  oops::Log::trace() << classname() << " readStructuredFields done" << std::endl;
}

// -------------------------------------------------------------------------------------------------

void IOStructuredGrid::print(std::ostream & os) const {
  os << classname() << " IO using Atlas Structured Grid";
}

// -------------------------------------------------------------------------------------------------

}  // namespace fv3jedi
