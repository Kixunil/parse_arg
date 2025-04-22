# Significant changes to the crate

## 1.0.1

* Removed broken 128-bit atomics - this is not breaking because the crate was always broken on targets that had 128-bit atomics in nightly
* Added `rust-version` field to manifest

## 1.0.0

* Deleted the helper functions to keep the crate focused

## 0.1.6

* Semver trick to depend on 1.0.0

## 0.1.5

* Various documentation and internal cleanups
* Implemented a bunch of `std` types: pointers, cells, locks, integers, `char`
* Implemented `std::error::Error` for `ParseArgError`

## 0.1.4

* Implemented `ParseArg` for `std::net::IpAddr`

## 0.1.3

* Added the `iter_short` function to help parsing short arguments

## 0.1.2

* Implemented `ParseArg` for `std::net::SocketAddr`

## 0.1.1

* Added helper function for parsing `--foo=bar` arguments
* Added `map_or` method to `ValueError`

## 0.1

Initial release
