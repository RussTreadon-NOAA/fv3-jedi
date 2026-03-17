! (C) Copyright 2017- UCAR
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.


module fv3jedi_tile_comms_mod

! mpi fortran
use mpi

! fv3jedi uses
use fv3jedi_geom_mod,   only: fv3jedi_geom
use fv3jedi_kinds_mod,  only: kind_real


! --------------------------------------------------------------------------------------------------


implicit none
private
public fv3jedi_tile_comms


! --------------------------------------------------------------------------------------------------


type fv3jedi_tile_comms

  integer :: isc, iec, jsc, jec, npx, npy, npxm1, npym1, npx_l, npy_l
  integer, allocatable :: isc_l(:), iec_l(:), jsc_l(:), jec_l(:)
  integer, allocatable :: vectorcounts(:), vectordispls(:)
  integer :: tcomm, tsize, trank

  contains
    procedure :: create
    procedure :: delete
    procedure, private :: scatter_tile_r2
    procedure, private :: scatter_tile_r3
    procedure, private :: gather_tile_r2
    procedure, private :: gather_tile_r3
    generic :: scatter_tile => scatter_tile_r2, scatter_tile_r3
    generic :: gather_tile => gather_tile_r2, gather_tile_r3
    procedure :: get_rank
    final     :: dummy_final

end type fv3jedi_tile_comms


! Description:
! This module provides a framework for handling MPI gatherig and scattering across individual tiles.
! Essentially one processor on each tile can gather or scatter the entire field for that tile to
! the other processors on that tile.

! --------------------------------------------------------------------------------------------------


contains


! --------------------------------------------------------------------------------------------------


subroutine create(self, geom)

! Args
class(fv3jedi_tile_comms), intent(inout) :: self
type(fv3jedi_geom),        intent(in)    :: geom

! Locals
integer :: ierr, n
integer, allocatable :: counts(:), displs(:)

! This produces 6 communicators, one for each tile of the cubed sphere
call mpi_comm_split(geom%f_comm%communicator(), geom%ntile, geom%f_comm%rank(), self%tcomm, ierr)

! Get the size and ranks for the new communicator
call mpi_comm_size(self%tcomm, self%tsize, ierr)
call mpi_comm_rank(self%tcomm, self%trank, ierr)

! Store local array sizes
self%isc = geom%isc
self%iec = geom%iec
self%jsc = geom%jsc
self%jec = geom%jec
self%npx = geom%npx
self%npy = geom%npy
self%npxm1 = geom%npx-1
self%npym1 = geom%npy-1
self%npx_l = geom%iec-geom%isc+1
self%npy_l = geom%jec-geom%jsc+1

! Arrays to hold dimensions of each patch on the tile
allocate(self%isc_l(self%tsize))
allocate(self%iec_l(self%tsize))
allocate(self%jsc_l(self%tsize))
allocate(self%jec_l(self%tsize))

!Array of counts and displacement
allocate(counts(self%tsize), displs(self%tsize))
do n = 1, self%tsize
   displs(n) = n-1
   counts(n) = 1
enddo

! Gather the local dimensions of each patch
call mpi_allgatherv(self%isc, 1, mpi_int, self%isc_l, counts, displs, mpi_int, self%tcomm, ierr)
call mpi_allgatherv(self%iec, 1, mpi_int, self%iec_l, counts, displs, mpi_int, self%tcomm, ierr)
call mpi_allgatherv(self%jsc, 1, mpi_int, self%jsc_l, counts, displs, mpi_int, self%tcomm, ierr)
call mpi_allgatherv(self%jec, 1, mpi_int, self%jec_l, counts, displs, mpi_int, self%tcomm, ierr)

! Allocate and set vector counts (to be later adjusted with knowlege of field_npz)
allocate(self%vectorcounts(self%tsize))
allocate(self%vectordispls(self%tsize))

! Deallocate locals
deallocate(counts)
deallocate(displs)

end subroutine create


! --------------------------------------------------------------------------------------------------


subroutine delete(self)

! Args
class(fv3jedi_tile_comms), intent(inout) :: self

! Locals
integer :: ierr

! Free the communicator
! ---------------------
call MPI_Comm_free(self%tcomm, ierr)

! Deallocate arrays
! -----------------
deallocate(self%isc_l)
deallocate(self%iec_l)
deallocate(self%jsc_l)
deallocate(self%jec_l)
deallocate(self%vectorcounts)
deallocate(self%vectordispls)


end subroutine delete


! --------------------------------------------------------------------------------------------------


integer function get_rank(self)

! Args
class(fv3jedi_tile_comms), intent(in) :: self

! Locals
integer :: ierr

! Return the rank of the current processor in the tile communicator
get_rank = self%trank

end function get_rank


! --------------------------------------------------------------------------------------------------


subroutine scatter_tile_r2(self, field_tile, field_patch)

! Arguments
class(fv3jedi_tile_comms), intent(inout) :: self
real(kind=kind_real),      intent(in)    :: field_tile(:, :)
real(kind=kind_real),      intent(inout) :: field_patch(self%isc:self%iec, self%jsc:self%jec)

! Locals
real(kind=kind_real), allocatable :: field_patch_r3(:,:,:)
real(kind=kind_real), allocatable :: field_tile_r3(:,:,:)

! Allocate and copy field_patch into rank 3 array
allocate(field_patch_r3(self%iec-self%isc+1, self%jec-self%jsc+1, 1))

! Allocate the passing array on the root procs
if (self%trank == 0) then
  allocate(field_tile_r3(self%npxm1, self%npym1, 1))
  field_tile_r3(:,:,1) = field_tile
endif

! Call scatter for the r3 arrays
call scatter_tile_r3(self, 1, field_tile_r3, field_patch_r3)

! Reshape result to 2D
field_patch = field_patch_r3(:,:,1)

end subroutine scatter_tile_r2


! --------------------------------------------------------------------------------------------------


subroutine scatter_tile_r3(self, field_npz, field_tile, field_patch)

! Arguments
class(fv3jedi_tile_comms), intent(inout) :: self
integer,                   intent(in)    :: field_npz
real(kind=kind_real),      intent(in)    :: field_tile(:, :, :)
real(kind=kind_real),      intent(inout) :: field_patch(self%isc:self%iec, self%jsc:self%jec, &
                                                        1:field_npz)

! Locals
integer :: ierr, n, jc, jk, jj, ji
real(kind=kind_real), allocatable :: vector_g(:), vector_l(:)


! Allocate receieving array (on all processors)
! ---------------------------------------------
allocate(vector_l(self%npx_l*self%npy_l*field_npz))


! Pack whole tile array into vector (on sending processors)
! ---------------------------------------------------------
if (self%trank == 0) then
  allocate(vector_g(self%npxm1*self%npym1*field_npz))
  n = 0
  do jc = 1, self%tsize
    self%vectordispls(jc) = n
    do jk = 1, field_npz
      do jj = self%jsc_l(jc), self%jec_l(jc)
        do ji = self%isc_l(jc), self%iec_l(jc)
          n = n+1
          vector_g(n) = field_tile(ji, jj, jk)
        enddo
      enddo
    enddo
    self%vectorcounts(jc) = n - self%vectordispls(jc)
  enddo
endif


! Scatter tile array to processors
! --------------------------------
call mpi_scatterv( vector_g, self%vectorcounts, self%vectordispls, mpi_double_precision, &
                   vector_l, self%npx_l*self%npy_l*field_npz, mpi_double_precision, &
                   0, self%tcomm, ierr )


! Unpack local vector into array
! ------------------------------
n = 0
do jk = 1, field_npz
  do jj = self%jsc, self%jec
    do ji = self%isc, self%iec
      n = n+1
      field_patch(ji, jj, jk) = vector_l(n)
    enddo
  enddo
enddo


! Deallocate locals
! -----------------
if (self%trank == 0) deallocate(vector_g)
deallocate(vector_l)


end subroutine scatter_tile_r3


! --------------------------------------------------------------------------------------------------


subroutine gather_tile_r2(self, field_patch, field_tile)

! Arguments
class(fv3jedi_tile_comms), intent(inout) :: self
real(kind=kind_real),      intent(in)    :: field_patch(self%isc:self%iec, self%jsc:self%jec)
real(kind=kind_real),      intent(inout) :: field_tile(:, :)

! Locals
real(kind=kind_real), allocatable :: field_tile_r3(:,:,:)

! Allocate and copy field_tile into rank 3 array
if (self%trank == 0) allocate(field_tile_r3(self%npxm1, self%npym1, 1))

! Call rank 3 version
call self%gather_tile_r3(1, reshape(field_patch, (/self%iec-self%isc+1, self%jec-self%jsc+1, 1/)), &
                         field_tile_r3)

! Extract the rank 2 array
if (self%trank == 0) field_tile = field_tile_r3(:,:,1)

end subroutine gather_tile_r2


! --------------------------------------------------------------------------------------------------


subroutine gather_tile_r3(self, field_npz, field_patch, field_tile)

! Arguments
class(fv3jedi_tile_comms), intent(inout) :: self
integer,                   intent(in)    :: field_npz
real(kind=kind_real),      intent(in)    :: field_patch(self%isc:self%iec, self%jsc:self%jec, &
                                                        1:field_npz)
real(kind=kind_real),      intent(inout) :: field_tile(:, :, :)

! Locals
integer :: ierr, n, jc, jk, jj, ji
real(kind=kind_real), allocatable :: vector_g(:), vector_l(:)


! Allocate flattened arrays for scatter
! -------------------------------------
if (self%trank == 0) allocate(vector_g(self%npxm1*self%npym1*field_npz))
allocate(vector_l(self%npx_l*self%npy_l*field_npz))


!Gather counts and displacement
! -----------------------------
self%vectordispls(1) = 0
n = 0
do jc = 1, self%tsize
  self%vectorcounts(jc) = (self%jec_l(jc)-self%jsc_l(jc)+1) * &
                          (self%iec_l(jc)-self%isc_l(jc)+1) * &
                          field_npz
  self%vectordispls(jc) = n
  n = n + self%vectorcounts(jc)
enddo


! Pack local array into vector
! ----------------------------
n = 0
do jk = 1, field_npz
  do jj = self%jsc, self%jec
    do ji = self%isc, self%iec
      n = n+1
      vector_l(n) = field_patch(ji, jj, jk)
    enddo
  enddo
enddo


! Gather the full field
! ---------------------
call mpi_gatherv( vector_l, self%npx_l*self%npy_l*field_npz, mpi_double_precision, &
                  vector_g, self%vectorcounts, self%vectordispls, mpi_double_precision, &
                  0, self%tcomm, ierr)


! Unpack global vector into array
! -------------------------------
if (self%trank == 0) then
  n = 0
  do jc = 1, self%tsize
    do jk = 1, field_npz
      do jj = self%jsc_l(jc),self%jec_l(jc)
        do ji = self%isc_l(jc),self%iec_l(jc)
          n = n+1
          field_tile(ji, jj, jk) = vector_g(n)
        enddo
      enddo
    enddo
  enddo
endif

! Deallocate locals
! -----------------
if (self%trank == 0) deallocate(vector_g)
deallocate(vector_l)


end subroutine gather_tile_r3


! --------------------------------------------------------------------------------------------------


subroutine dummy_final(self)
type(fv3jedi_tile_comms), intent(inout) :: self
end subroutine dummy_final


! --------------------------------------------------------------------------------------------------


end module fv3jedi_tile_comms_mod
