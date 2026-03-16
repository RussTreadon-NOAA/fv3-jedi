/*
 * (C) Copyright 2025- UCAR.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#pragma once

#include <memory>
#include <ostream>
#include <string>

#include "oops/generic/GlobalInterpolator.h"

#include "oops/util/DateTime.h"
#include "oops/util/parameters/OptionalParameter.h"
#include "oops/util/parameters/Parameter.h"
#include "oops/util/parameters/Parameters.h"
#include "oops/util/parameters/RequiredParameter.h"

#include "fv3jedi/IO/Utils/IOBase.h"

namespace fv3jedi {

// -------------------------------------------------------------------------------------------------

class IOStructuredGridParameters : public IOParametersBase {
  OOPS_CONCRETE_PARAMETERS(IOStructuredGridParameters, IOParametersBase)

 public:
  // Type of structured grid to write
  oops::Parameter<std::string> outputGridType{"gridtype", "gridtype", "F12", this};

  // Filename of output (write path, or single-file read fallback)
  oops::Parameter<std::string> filename{"filename", "filename",
                                        "cube_to_geometric_%Y%m%dT%H%M%S.nc4", this};

  // Directory containing input files (used together with filenames list for reading)
  oops::Parameter<std::string> datapath{"datapath", "datapath", "", this};

  // Ordered list of input filenames for reading (fields are searched across all files)
  oops::Parameter<std::vector<std::string>> filenames{"filenames", "filenames", {}, this};

  // Interpolator type
  oops::Parameter<std::string> interpolator{"local interpolator type", "local interpolator type",
                                            "oops unstructured grid interpolator",
                                            this};

  // Optionally config may contain member
  oops::OptionalParameter<int> member{"member", "ensemble member number", this};

  // Floating point precision in bytes for NetCDF write
  oops::Parameter<int> floatPrecision{"float precision in bytes", "float precision in bytes", 8,
                                      this};

  // Dimension names used when writing
  oops::Parameter<std::string> latName{"latitude dim name", "latitude dim name", "lat", this};
  oops::Parameter<std::string> lonName{"longitude dim name", "longitude dim name", "lon", this};
  oops::Parameter<std::string> levName{"level dim name", "level dim name", "lev", this};
  oops::Parameter<std::string> edgName{"edge dim name", "edge dim name", "edge", this};
  oops::Parameter<std::string> forName{"four level dim name", "four level dim name", "four", this};
  oops::Parameter<std::string> timName{"time dim name", "time dim name", "time", this};

  // Dimension names used when reading (default to UFS/GFS model output conventions)
  oops::Parameter<std::string> readLatName{"read latitude dim name", "read latitude dim name",
                                           "grid_yt", this};
  oops::Parameter<std::string> readLonName{"read longitude dim name", "read longitude dim name",
                                           "grid_xt", this};

  // Latitude storage order in the input files.
  // false (default): latitudes stored north-to-south (nc_j=0 is northernmost), matching
  //                  UFS/GFS model output where grid_yt[0] ≈ +89°.
  // true:            latitudes stored south-to-north (nc_j=0 is southernmost), matching
  //                  files written by IOStructuredGrid::writeStructuredFields.
  oops::Parameter<bool> readLatSouthToNorth{"read lat south to north",
                                            "Whether input files store latitudes south-to-north"
                                            " (true) or north-to-south (false, UFS/GFS default)",
                                            false, this};
};

// -------------------------------------------------------------------------------------------------
class IOStructuredGrid : public IOBase, private util::ObjectCounter<IOStructuredGrid> {
 public:
  static const std::string classname() {return "fv3jedi::IOStructuredGrid";}

  typedef IOStructuredGridParameters Parameters_;

  IOStructuredGrid(const Geometry &, const Parameters_ &);
  ~IOStructuredGrid();
  void read(State &, const eckit::LocalConfiguration &,
            const eckit::LocalConfiguration &) const override;
  void read(Increment &, const eckit::LocalConfiguration &,
            const eckit::LocalConfiguration &) const override;
  void write(const State &, const eckit::LocalConfiguration &,
             const eckit::LocalConfiguration &) const override;
  void write(const Increment &, const eckit::LocalConfiguration &,
             const eckit::LocalConfiguration &) const override;

 private:
  // Methods
  void print(std::ostream &) const override;
  template <typename T>
  void interpAndWrite(const T & obj, const std::string & label,
                      const eckit::LocalConfiguration & fileionames,
                      const eckit::LocalConfiguration & fileioscaling) const;
  void writeStructuredFields(const atlas::FieldSet &, const util::DateTime &,
                             const eckit::LocalConfiguration &,
                             const eckit::LocalConfiguration &) const;

  template <typename T>
  void readAndInterp(T & obj, const std::string & label,
                     const eckit::LocalConfiguration & fileionames,
                     const eckit::LocalConfiguration & fileioscaling) const;
  void readStructuredFields(const util::DateTime &, atlas::FieldSet &,
                            const eckit::LocalConfiguration & ioNames) const;

  // Data
  std::unique_ptr<oops::GlobalInterpolator> interpolator_;
  std::unique_ptr<oops::GlobalInterpolator> readInterpolator_;
  const Geometry & geom_;
  std::string gridStr_;
  Parameters_ params_;
  std::unique_ptr<atlas::functionspace::StructuredColumns> writeFunctionSpace_;
  std::unique_ptr<atlas::functionspace::StructuredColumns> readFunctionSpace_;
};

// -------------------------------------------------------------------------------------------------

}  // namespace fv3jedi
