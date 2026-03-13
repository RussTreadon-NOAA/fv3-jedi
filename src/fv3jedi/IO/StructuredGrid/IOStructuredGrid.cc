/*
 * (C) Copyright 2025- UCAR.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#include <netcdf.h>

#include <map>
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
// -------------------------------------------------------------------------------------------------

IOStructuredGrid::IOStructuredGrid(const Geometry & geom, const Parameters_ & params)
  : IOBase(geom, params.toConfiguration()), params_(params), gridStr_(""), geom_(geom),
    writeFunctionSpace_(), writeInterpolator_(),
    readFunctionSpace_(), readInterpolator_() {
  util::Timer timer(classname(), "IOStructuredGrid");
  oops::Log::trace() << classname() << " constructor starting" << std::endl;

  // Determine the Atlas grid string
  // --------------------------------
  std::string outputGridType = params.outputGridType.value();

  // Convert legacy shorthand gridtype to what Atlas expects
  if (outputGridType == "latlon") {
    outputGridType = "L" + std::to_string(4*(geom.npx()-1)) + "x" +
                     std::to_string(2*(geom.npy()-1)+1);
  } else if (outputGridType == "gaussian") {
    outputGridType = "F" + std::to_string(geom.npy()-1);
  }

  if (outputGridType[0] == 'L') {
    gridStr_ = "latlon";
  } else if (outputGridType[0] == 'F') {
    gridStr_ = "gaussian";
  } else {
    ABORT("IOStructuredGrid: outputGridType must begin with L (latlon) or F (regular gaussian). ");
  }

  // Generate the Atlas grid object
  const atlas::Grid grid(outputGridType);

  // Atlas configuration with communicator name
  eckit::LocalConfiguration atlas_conf;
  atlas_conf.set("mpi_comm", geom.getComm().name());

  // --- Write function space: serial (all points on rank 0) ---
  {
    std::vector<int> zeros(grid.size(), 0);
    const atlas::grid::Distribution dist(geom.getComm().size(), grid.size(), zeros.data());
    writeFunctionSpace_.reset(new atlas::functionspace::StructuredColumns(grid, dist, atlas_conf));
  }

  // Write interpolator: cube sphere -> serial structured grid
  // Uses GeometryData built from the cube sphere function space (requires NodeColumns in GDASApp)
  {
    oops::GeometryData geomData(geom.functionSpace(), geom.fields(), geom.levelsAreTopDown(),
                                geom.getComm());
    writeInterpolator_.reset(new oops::GlobalInterpolator(params.toConfiguration(), geomData,
                                                          *writeFunctionSpace_,
                                                          geom.getComm()));
  }

  // --- Read function space: balanced distribution across all ranks ---
  // A balanced (non-serial) distribution ensures GeometryData can build its triangulation
  // (GeometryData skips setup when some MPI tasks own zero points, which happens with
  // the serial distribution used by writeFunctionSpace_).
  {
    readFunctionSpace_.reset(new atlas::functionspace::StructuredColumns(grid, atlas_conf));
  }

  // Read interpolator: balanced structured grid -> cube sphere
  {
    oops::GeometryData readGeomData(*readFunctionSpace_, atlas::FieldSet(),
                                    geom.levelsAreTopDown(), geom.getComm());
    readInterpolator_.reset(new oops::GlobalInterpolator(params.toConfiguration(), readGeomData,
                                                         geom.functionSpace(),
                                                         geom.getComm()));
  }

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

template <typename T>
void IOStructuredGrid::interpAndWrite(const T & obj, const std::string & label,
                                      const eckit::LocalConfiguration & fileionames,
                                      const eckit::LocalConfiguration & fileioscaling) const {
  util::Timer timer(classname(), "write " + label);
  oops::Log::trace() << classname() << " write " << label << " starting" << std::endl;

  // Get cube sphere fields and interpolate to serial structured grid
  atlas::FieldSet fieldsCubeSphere;
  atlas::FieldSet fieldsGeographic;
  obj.toFieldSet(fieldsCubeSphere);
  writeInterpolator_->apply(fieldsCubeSphere, fieldsGeographic);

  // Only rank 0 holds all structured grid data (serial distribution) - write to file
  if (geom_.getComm().rank() == 0) {
    this->writeStructuredFields(fieldsGeographic, obj.validTime(), fileionames, fileioscaling);
  }

  oops::Log::trace() << classname() << " write " << label << " done" << std::endl;
}

// -------------------------------------------------------------------------------------------------

template <typename T>
void IOStructuredGrid::readAndInterp(T & obj, const std::string & label,
                                     const eckit::LocalConfiguration & fileionames,
                                     const eckit::LocalConfiguration & fileioscaling) const {
  util::Timer timer(classname(), "read " + label);
  oops::Log::trace() << classname() << " read " << label << " starting" << std::endl;

  // Get cube sphere fields structure (provides field names and level counts)
  atlas::FieldSet fieldsCubeSphere;
  obj.toFieldSet(fieldsCubeSphere);

  // Build a structured grid FieldSet on the balanced readFunctionSpace_
  // with the same fields as the cube sphere
  atlas::FieldSet fieldsGeographic;
  for (const auto & cubeField : fieldsCubeSphere) {
    fieldsGeographic.add(readFunctionSpace_->createField<double>(
        atlas::option::name(cubeField.name()) |
        atlas::option::levels(cubeField.shape(1))));
  }

  // Read structured grid data from NetCDF; each rank reads its local rows
  this->readStructuredFields(obj.validTime(), fieldsGeographic, fileionames);

  // Interpolate from balanced structured grid to distributed cube sphere
  readInterpolator_->apply(fieldsGeographic, fieldsCubeSphere);

  // Update the state/increment from the cube sphere fields
  obj.fromFieldSet(fieldsCubeSphere);

  oops::Log::trace() << classname() << " read " << label << " done" << std::endl;
}

// -------------------------------------------------------------------------------------------------

static inline void nc_rc(const int return_code, const std::string & operation) {
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
  int fileId;
  int latId, lonId, levId, edgId, forId, timId;
  int fIv;
  std::map<std::string, int> fieldIvs;

  // Get ak/bk for writing as global attributes
  std::vector<double> ak = geom_.ak();
  std::vector<double> bk = geom_.bk();

  // Format the filename with datetime
  std::string pathFile = params_.filename.value();
  if (pathFile.find("%Y") == std::string::npos) {
    pathFile += "%Y%m%d_%H%M%Sz";
  }
  if (pathFile.find(".nc") == std::string::npos) {
    pathFile += ".nc4";
  }
  pathFile = time.formatString(pathFile);
  util::stringfunctions::swapNameMember(params_.toConfiguration(), pathFile);

  // Create file
  nc_rc(nc_create(pathFile.c_str(), NC_CLOBBER | NC_NETCDF4, &fileId), "nc_create " + pathFile);

  // Grid dimensions
  const atlas::RegularGrid regGrid(writeFunctionSpace_->grid());
  const int nLat = regGrid.ny();
  const int nLon = regGrid.nx();
  const int nLev = geom_.npz();
  const int nEdg = nLev + 1;
  const int nFor = 4;
  const int nTim = 1;

  nc_rc(nc_def_dim(fileId, params_.latName.value().c_str(), nLat, &latId), "nc_def_dim (lat)");
  nc_rc(nc_def_dim(fileId, params_.lonName.value().c_str(), nLon, &lonId), "nc_def_dim (lon)");
  nc_rc(nc_def_dim(fileId, params_.levName.value().c_str(), nLev, &levId), "nc_def_dim (lev)");
  nc_rc(nc_def_dim(fileId, params_.edgName.value().c_str(), nEdg, &edgId), "nc_def_dim (edg)");
  nc_rc(nc_def_dim(fileId, params_.forName.value().c_str(), nFor, &forId), "nc_def_dim (for)");
  nc_rc(nc_def_dim(fileId, params_.timName.value().c_str(), nTim, &timId), "nc_def_dim (tim)");

  // Coordinate arrays (lat stored south-to-north, matching global convention)
  std::vector<double> latArr(nLat), lonArr(nLon);
  std::vector<int> levArr(nLev), edgArr(nEdg), forArr(nFor), timArr(nTim);
  for (int i = 0; i < nLat; ++i) { latArr[i] = regGrid.y(nLat - 1 - i); }
  for (int i = 0; i < nLon; ++i) { lonArr[i] = regGrid.x(i); }
  for (int i = 0; i < nLev; ++i) { levArr[i] = i + 1; }
  for (int i = 0; i < nEdg; ++i) { edgArr[i] = i + 1; }
  for (int i = 0; i < nFor; ++i) { forArr[i] = i + 1; }
  for (int i = 0; i < nTim; ++i) { timArr[i] = i + 1; }

  // Define coordinate variables
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
  nc_rc(nc_put_att_text(fileId, fIv, "units", strlen("1"), "1"), "nc_put_att_text (lev)");
  fieldIvs[params_.levName.value()] = fIv;

  nc_rc(nc_def_var(fileId, params_.edgName.value().c_str(), NC_INT, 1, &edgId, &fIv),
        "nc_def_var (edg)");
  nc_rc(nc_put_att_text(fileId, fIv, "units", strlen("1"), "1"), "nc_put_att_text (edg)");
  fieldIvs[params_.edgName.value()] = fIv;

  nc_rc(nc_def_var(fileId, params_.forName.value().c_str(), NC_INT, 1, &forId, &fIv),
        "nc_def_var (for)");
  nc_rc(nc_put_att_text(fileId, fIv, "units", strlen("1"), "1"), "nc_put_att_text (for)");
  fieldIvs[params_.forName.value()] = fIv;

  nc_rc(nc_def_var(fileId, params_.timName.value().c_str(), NC_INT, 1, &timId, &fIv),
        "nc_def_var (tim)");
  nc_rc(nc_put_att_text(fileId, fIv, "units", strlen("1"), "1"), "nc_put_att_text (tim)");
  fieldIvs[params_.timName.value()] = fIv;

  // Dimension sets per level count
  std::map<int, std::vector<int>> fieldDims;
  fieldDims[nLev] = {timId, levId, latId, lonId};
  fieldDims[nEdg] = {timId, edgId, latId, lonId};
  fieldDims[4]    = {timId, forId, latId, lonId};
  fieldDims[1]    = {timId, latId, lonId};
  fieldDims[0]    = {timId, latId, lonId};

  const int floatPrecision = params_.floatPrecision.value();
  const int ncPrec = (floatPrecision == 4) ? NC_FLOAT : NC_DOUBLE;

  // Define field variables
  for (auto & field : fields) {
    const int nLevField = field.shape(1);
    auto it = fieldDims.find(nLevField);
    if (it == fieldDims.end()) {
      std::ostringstream oss;
      oss << "IOStructuredGrid::writeStructuredFields: "
          << "No entry in fieldDims for field '" << field.name()
          << "' with " << nLevField << " levels.";
      ABORT(oss.str());
    }
    const auto & dims = it->second;
    const std::string fieldLong = field.name();
    const FieldMetadata & fieldMetadata = geom_.fieldsMetaData().getFieldMetadata(fieldLong);
    std::string unitsStr = fieldMetadata.getVarUnits();

    // Allow ioNames to remap the field variable name in the file
    std::string fieldName = fieldLong;
    if (ioNames.has(fieldLong)) {
      fieldName = ioNames.getString(fieldLong);
    }

    nc_rc(nc_def_var(fileId, fieldName.c_str(), ncPrec, dims.size(), dims.data(), &fIv),
          "nc_def_var " + fieldName);
    nc_rc(nc_put_att_text(fileId, fIv, "units", strlen(unitsStr.c_str()), unitsStr.c_str()),
          "nc_put_att_text " + fieldName + " units");
    nc_rc(nc_put_att_text(fileId, fIv, "long_name", strlen(fieldLong.c_str()), fieldLong.c_str()),
          "nc_put_att_text " + fieldName + " long_name");
    fieldIvs[field.name()] = fIv;
  }

  // Global attributes: ak/bk (hybrid pressure coords), grid type, dimensions
  if (!ak.empty()) {
    nc_rc(nc_put_att_double(fileId, NC_GLOBAL, "ak", NC_DOUBLE, ak.size(), ak.data()),
          "nc_put_att_double (ak)");
  }
  if (!bk.empty()) {
    nc_rc(nc_put_att_double(fileId, NC_GLOBAL, "bk", NC_DOUBLE, bk.size(), bk.data()),
          "nc_put_att_double (bk)");
  }
  nc_rc(nc_put_att_text(fileId, NC_GLOBAL, "grid", strlen(gridStr_.c_str()), gridStr_.c_str()),
        "nc_put_att_text (grid)");
  nc_rc(nc_put_att_int(fileId, NC_GLOBAL, "im", NC_INT, 1, &nLon), "nc_put_att_int (im)");
  nc_rc(nc_put_att_int(fileId, NC_GLOBAL, "jm", NC_INT, 1, &nLat), "nc_put_att_int (jm)");

  nc_rc(nc_enddef(fileId), "nc_enddef");

  // Write coordinate data
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

  // Write field data
  // Atlas StructuredColumns (serial): point index = j_atlas * nLon + i
  // where j_atlas=0 is northernmost.  File stores lat south-to-north:
  //   file j_nc = nLat-1-j_atlas  =>  j_atlas = nLat-1-j_nc
  // So values[k*nLat*nLon + j_nc*nLon + i] = fieldView[(nLat-1-j_nc)*nLon + i, k]
  for (auto & field : fields) {
    const int nLevField = field.shape(1);
    const auto fieldView = atlas::array::make_view<double, 2>(field);
    std::vector<double> values(nLat * nLon * nLevField);
    for (int k = 0; k < nLevField; ++k) {
      for (int j = 0; j < nLat; ++j) {
        for (int i = 0; i < nLon; ++i) {
          values[k*nLat*nLon + j*nLon + i] = fieldView((nLat - 1 - j) * nLon + i, k);
        }
      }
    }
    nc_rc(nc_put_var_double(fileId, fieldIvs[field.name()], values.data()),
          "nc_put_var_double " + field.name());
  }

  nc_rc(nc_close(fileId), "nc_close");
}

// -------------------------------------------------------------------------------------------------

void IOStructuredGrid::readStructuredFields(const util::DateTime & time,
                                            atlas::FieldSet & fields,
                                            const eckit::LocalConfiguration & ioNames) const {
  // Format the filename with datetime (same logic as write)
  std::string pathFile = params_.filename.value();
  if (pathFile.find("%Y") == std::string::npos) {
    pathFile += "%Y%m%d_%H%M%Sz";
  }
  if (pathFile.find(".nc") == std::string::npos) {
    pathFile += ".nc4";
  }
  pathFile = time.formatString(pathFile);
  util::stringfunctions::swapNameMember(params_.toConfiguration(), pathFile);

  // Open for reading (all ranks open in parallel; NC_NOWRITE allows concurrent reads)
  int fileId;
  nc_rc(nc_open(pathFile.c_str(), NC_NOWRITE, &fileId), "nc_open " + pathFile);

  // Read grid dimensions from the file
  int latDimId, lonDimId;
  size_t nLat, nLon;
  nc_rc(nc_inq_dimid(fileId, params_.latName.value().c_str(), &latDimId),
        "nc_inq_dimid (lat)");
  nc_rc(nc_inq_dimid(fileId, params_.lonName.value().c_str(), &lonDimId),
        "nc_inq_dimid (lon)");
  nc_rc(nc_inq_dimlen(fileId, latDimId, &nLat), "nc_inq_dimlen (lat)");
  nc_rc(nc_inq_dimlen(fileId, lonDimId, &nLon), "nc_inq_dimlen (lon)");

  // Coordinate index convention used throughout this method:
  //
  //   Atlas StructuredColumns j index: j=0 is NORTHERNMOST latitude.
  //   NetCDF file j index (nc_j):       nc_j=0 is SOUTHERNMOST latitude (south-to-north storage).
  //   Relationship:  nc_j = nLat - 1 - j_atlas  <=>  j_atlas = nLat - 1 - nc_j
  //
  // This rank owns Atlas rows [j_beg, j_end).  The corresponding contiguous NetCDF row block is
  //   nc_j in [nLat - j_end,  nLat - 1 - j_beg]
  // which starts at nc_j_start = nLat - j_end.
  //
  // The read buffer for a multi-level field is ordered (lev, nc_j_local, lon) where
  // nc_j_local = 0 corresponds to nc_j = nc_j_start (southernmost row in this rank's block),
  // i.e., nc_j_local = nc_j - nc_j_start = (nLat - 1 - j_atlas) - (nLat - j_end)
  //                  = j_end - 1 - j_atlas.
  const int j_beg = readFunctionSpace_->j_begin();
  const int j_end = readFunctionSpace_->j_end();
  const int myNLat = j_end - j_beg;
  const size_t nc_j_start = static_cast<size_t>(nLat) - static_cast<size_t>(j_end);

  for (auto & field : fields) {
    const std::string fieldLong = field.name();
    const int nLevField = field.shape(1);

    // Allow ioNames to remap the field variable name in the file (symmetrical with write).
    std::string fieldName = fieldLong;
    if (ioNames.has(fieldLong)) {
      fieldName = ioNames.getString(fieldLong);
    }

    // Look up the variable in the file
    int varId;
    const int rc = nc_inq_varid(fileId, fieldName.c_str(), &varId);
    if (rc != NC_NOERR) {
      oops::Log::warning() << classname() << "::readStructuredFields: field '"
                           << fieldName << "' not found in " << pathFile
                           << " -- leaving at zero." << std::endl;
      continue;
    }

    // Read only this rank's rows.
    // File variable dims:
    //   surface (nLevField==1): (time, lat, lon)        -- 3-D variable
    //   multi-level           : (time, lev/edge/four, lat, lon)  -- 4-D variable
    std::vector<double> values;
    if (nLevField == 1) {
      // 3D variable: (time=1, lat=nLat, lon=nLon)
      std::vector<size_t> start = {0, nc_j_start, 0};
      std::vector<size_t> count = {1, static_cast<size_t>(myNLat), nLon};
      values.resize(myNLat * static_cast<int>(nLon));
      nc_rc(nc_get_vara_double(fileId, varId, start.data(), count.data(), values.data()),
            "nc_get_vara_double " + fieldName);
    } else {
      // 4D variable: (time=1, lev=nLevField, lat=nLat, lon=nLon)
      std::vector<size_t> start = {0, 0, nc_j_start, 0};
      std::vector<size_t> count = {1, static_cast<size_t>(nLevField),
                                   static_cast<size_t>(myNLat), nLon};
      values.resize(nLevField * myNLat * static_cast<int>(nLon));
      nc_rc(nc_get_vara_double(fileId, varId, start.data(), count.data(), values.data()),
            "nc_get_vara_double " + fieldName);
    }

    // Fill the Atlas StructuredColumns field view using the index convention described above.
    // Buffer index: values[k * myNLat * nLon + nc_j_local * nLon + i]
    //   where nc_j_local = j_end - 1 - j_atlas  (see comment block above).
    auto fieldView = atlas::array::make_view<double, 2>(field);
    for (int j = j_beg; j < j_end; ++j) {
      const int nc_j_local = (j_end - 1) - j;
      for (atlas::idx_t i = readFunctionSpace_->i_begin(j);
           i < readFunctionSpace_->i_end(j); ++i) {
        const atlas::idx_t localIdx = readFunctionSpace_->index(j, i);
        for (int k = 0; k < nLevField; ++k) {
          fieldView(localIdx, k) =
              values[k * myNLat * static_cast<int>(nLon)
                     + nc_j_local * static_cast<int>(nLon) + i];
        }
      }
    }
    field.set_dirty();
  }

  nc_rc(nc_close(fileId), "nc_close");
}

// -------------------------------------------------------------------------------------------------

void IOStructuredGrid::print(std::ostream & os) const {
  os << classname() << " IO using Atlas Structured Grid";
}

// -------------------------------------------------------------------------------------------------

}  // namespace fv3jedi
