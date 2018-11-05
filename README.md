Parse arg
=========

Traits and implementations for parsing command-line arguments.

About
-----

This crate provides traits and implementations for parsing command-line arguments.
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

Contibutions
------------

I will merge new implementations liberally, provided these conditions are met:

* The type is either in `std` or in another crate which is added as an *optional* dependency.
* The description of the type is gramatically correct, readable explanation of the type, that
  can be appended to the string "The input must be " without sounding unnatural.
* The user must be able to know how to formt the input based on the description without any
  googling. In case of doubt provide more precise description in parentheses.
* Your contribution is licensed under MIT or MITNFA license.

License
-------

MITNFA
