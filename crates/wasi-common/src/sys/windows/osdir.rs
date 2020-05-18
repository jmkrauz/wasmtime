use super::oshandle::RawOsHandle;
use crate::handle::HandleRights;
use crate::wasi::{types, RightsExt};
use std::cell::Cell;
use std::convert::TryFrom;
use std::fs::File;
use std::io;
use std::os::windows::prelude::{AsRawHandle, FromRawHandle, IntoRawHandle};

#[derive(Debug)]
pub(crate) struct OsDir {
    pub(crate) rights: Cell<HandleRights>,
    pub(crate) handle: RawOsHandle,
}

impl OsDir {
    pub(crate) fn new(rights: HandleRights, handle: RawOsHandle) -> io::Result<Self> {
        let rights = Cell::new(rights);
        Ok(Self { rights, handle })
    }
}

impl TryFrom<File> for OsDir {
    type Error = io::Error;

    fn try_from(file: File) -> io::Result<Self> {
        let ft = file.metadata()?.file_type();
        if !ft.is_dir() {
            return Err(io::Error::from_raw_os_error(libc::EINVAL));
        }
        let rights = get_rights(&file)?;
        let handle = unsafe { RawOsHandle::from_raw_handle(file.into_raw_handle()) };
        Self::new(rights, handle)
    }
}

fn get_rights(file: &File) -> io::Result<HandleRights> {
    use winx::file::{query_access_information, AccessMode};
    let mut rights = HandleRights::new(
        types::Rights::directory_base(),
        types::Rights::directory_inheriting(),
    );
    let mode = query_access_information(file.as_raw_handle())?;
    if mode.contains(AccessMode::FILE_GENERIC_READ) {
        rights.base |= types::Rights::FD_READ;
    }
    if mode.contains(AccessMode::FILE_GENERIC_WRITE) {
        rights.base |= types::Rights::FD_WRITE;
    }
    Ok(rights)
}
