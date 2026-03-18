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
#include <vector>

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

  // Filename of output (write / backward-compatible single-file read)
  oops::Parameter<std::string> filename{"filename", "filename",
                                        "cube_to_geometric_%Y%m%dT%H%M%S.nc4", this};

  // Optional path prefix applied to every entry in filenames (read use-case)
  oops::Parameter<std::string> datapath{"datapath", "datapath", "", this};

  // List of input files for multi-file read (e.g. separate atm and sfc files).
  // When non-empty, takes precedence over filename for read operations.
  // Duplicate variable names across files are resolved with a first-file-wins
  // policy; a trace log is emitted when a field is skipped due to duplication.
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

  // Dimension names (write)
  oops::Parameter<std::string> latName{"latitude dim name", "latitude dim name", "lat", this};
  oops::Parameter<std::string> lonName{"longitude dim name", "longitude dim name", "lon", this};
  oops::Parameter<std::string> levName{"level dim name", "level dim name", "lev", this};
  oops::Parameter<std::string> edgName{"edge dim name", "edge dim name", "edge", this};
  oops::Parameter<std::string> forName{"four level dim name", "four level dim name", "four", this};
  oops::Parameter<std::string> timName{"time dim name", "time dim name", "time", this};

  // Dimension names (read) — UFS/GFS input files use grid_yt / grid_xt
  oops::Parameter<std::string> readLatName{"read latitude dim name",
                                           "read latitude dim name", "grid_yt", this};
  oops::Parameter<std::string> readLonName{"read longitude dim name",
                                           "read longitude dim name", "grid_xt", this};
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
  void readStructuredFields(atlas::FieldSet &, const std::vector<std::string> &,
                            const util::DateTime &,
                            const eckit::LocalConfiguration &,
                            util::DateTime * fileTime = nullptr) const;
  atlas::Field readVarToStructuredAtlasField(int fileId, const std::string & varName,
                                             int nLat, int nLon,
                                             const std::string & latDimName,
                                             const std::string & lonDimName,
                                             bool flipJ) const;

  // Data
  std::unique_ptr<oops::GlobalInterpolator> interpolator_;
  // readInterpolator_ and readFunctionSpace_ are mutable because they are lazily updated
  // in readStructuredFields (a const method) when the input file's Gaussian grid differs
  // from the one created in the constructor.  This follows the standard lazy-init pattern.
  mutable std::unique_ptr<oops::GlobalInterpolator> readInterpolator_;
  const Geometry & geom_;
  std::string gridStr_;
  Parameters_ params_;
  std::unique_ptr<atlas::functionspace::StructuredColumns> writeFunctionSpace_;
  mutable std::unique_ptr<atlas::functionspace::StructuredColumns> readFunctionSpace_;
};

// -------------------------------------------------------------------------------------------------

}  // namespace fv3jedi
