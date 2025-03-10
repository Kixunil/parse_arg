Parse arg
=========

Traits and implementations for parsing command-line arguments.

About
-----

This crate provides traits to bridge various libraries providing parsable types with libraries
providing command line parsing implementations.
The core of the crate is `ParseArg` trait. It works much like `FromStr` trait, but with
these differences:

* It operates on `&OsStr` instead of `&str`, thus allowing wider range of possible inputs.
* It provides `parse_owned_arg()` method which can be specialized to avoid allocations.
* It requires the implementor to provide `describe_type()` to print human-readable description.
  of expected input.
* It requires the error type to implement `Display` in order to enable user-friendly interface.

Further, the crate provides `ParseArgFromStr` trait, which causes any type implementing it to
auto-implement `ParseArg` trait. This is handy when implementing `ParseArg` for types that
already have `FromStr` implemented, so that boilerplate is reduced.

Any libraries that wish to help their consumers implement parsing their types from command line
may add this crate as an optional dependency and implement the `ParseArg` trait (directly or
indirectly) for their types.

Any binaries wishing to use these traits should enable the `parse_arg` feature of the
librariess that use this crate and use a CLI parses implementation crate that uses it too.
Currently the only known implementation is [`configure_me`](https://docs.rs/configure_me) which
is also capable of parsing configuration files.

Maintenance status
------------------

This crate is considered done. There shouldn't be any updates to it required, so there probably
won't be any changes from now. I will accept PRs that fix problems but I don't expect to actively
search for things to improve. The crate has only two traits and one enum, so this should be
completely fine to use even if it becomes old.

MSRV
----

The minimum supported Rust version of the crate is 1.63 and will always be whichever Rust
version the current Debian stable (12 - Bookworm at the time of writing) supports.

License
-------

MITNFA
