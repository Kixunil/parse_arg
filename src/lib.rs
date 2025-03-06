//! This crate provides traits to bridge various libraries providing parsable types with libraries
//! providing command line parsing implementations.
//! The core of the crate is `ParseArg` trait. It works much like `FromStr` trait, but with
//! these differences:
//! 
//! * It operates on `&OsStr` instead of `&str`, thus allowing wider range of possible inputs.
//! * It provides `parse_owned_arg()` method which can be specialized to avoid allocations.
//! * It requires the implementor to provide `describe_type()` to print human-readable description.
//!   of expected input.
//! * It requires the error type to implement `Display` in order to enable user-friendly interface.
//! 
//! Further, the crate provides `ParseArgFromStr` trait, which causes any type implementing it to
//! auto-implement `ParseArg` trait. This is handy when implementing `ParseArg` for types that
//! already have `FromStr` implemented, so that boilerplate is reduced.
//!
//! Any libraries that wish to help their consumers implement parsing their types from command line
//! may add this crate as an optional dependency and implement the `ParseArg` trait (directly or
//! indirectly) for their types.
//!
//! Any binaries wishing to use these traits should enable the `parse_arg` feature of the
//! librariess that use this crate and use a CLI parses implementation crate that uses it too.
//! Currently the only known implementation is [`configure_me`](https://docs.rs/configure_me) which
//! is also capable of parsing configuration files.
//!
//! # MSRV
//!
//! The minimum supported Rust version of the crate is 1.63 and will always be whichever Rust
//! version the current Debian stable (12 - Bookworm at the time of writing) supports.

#![deny(missing_docs)]

use std::fmt;
use std::str::FromStr;
use std::ffi::{OsStr, OsString};

/// Defines an interface for types that can be created by parsing command-line argument.
///
/// This trait is similar to `FromStr`. See the crate documentation for list of important
/// differences.
pub trait ParseArg: Sized {
    /// Type returned in the `Err` variant of `Result` when parsing fails.
    type Error: fmt::Display;

    /// Parses the argument.
    fn parse_arg(arg: &OsStr) -> Result<Self, Self::Error>;

    /// Writes human-readable description of the type to the writer.
    ///
    /// The description should be in English composed in such way that appending it to string "The
    /// input must be " sounds natural. E.g. if the description is "a number", the resulting phrase
    /// will be "The input must be a number".
    ///
    /// This way, it can be used as a documentation/hint for the user.
    fn describe_type<W: fmt::Write>(writer: W) -> fmt::Result;

    /// Parses the argument consuming it.
    ///
    /// Implementors are encouraged to specialize this method if the resulting implementation is
    /// more performant - e.g. if it avoids allocation.
    ///
    /// The users are encouraged to use this method instead of `parse_arg` if they own the string
    /// and will not need it after call to this function. (Typical when working with
    /// `std::env::args_os()`.)
    fn parse_owned_arg(arg: OsString) -> Result<Self, Self::Error> {
        Self::parse_arg(&arg)
    }
}

impl<T: ParseArg> ParseArg for std::cell::Cell<T> {
    type Error = T::Error;

    fn parse_arg(arg: &OsStr) -> Result<Self, Self::Error> {
        T::parse_arg(arg).map(Into::into)
    }

    fn describe_type<W: fmt::Write>(writer: W) -> fmt::Result {
        T::describe_type(writer)
    }

    fn parse_owned_arg(arg: OsString) -> Result<Self, Self::Error> {
        T::parse_owned_arg(arg).map(Into::into)
    }
}

impl<T: ParseArg> ParseArg for std::cell::RefCell<T> {
    type Error = T::Error;

    fn parse_arg(arg: &OsStr) -> Result<Self, Self::Error> {
        T::parse_arg(arg).map(Into::into)
    }

    fn describe_type<W: fmt::Write>(writer: W) -> fmt::Result {
        T::describe_type(writer)
    }

    fn parse_owned_arg(arg: OsString) -> Result<Self, Self::Error> {
        T::parse_owned_arg(arg).map(Into::into)
    }
}

impl<T: ParseArg> ParseArg for std::cell::UnsafeCell<T> {
    type Error = T::Error;

    fn parse_arg(arg: &OsStr) -> Result<Self, Self::Error> {
        T::parse_arg(arg).map(Into::into)
    }

    fn describe_type<W: fmt::Write>(writer: W) -> fmt::Result {
        T::describe_type(writer)
    }

    fn parse_owned_arg(arg: OsString) -> Result<Self, Self::Error> {
        T::parse_owned_arg(arg).map(Into::into)
    }
}

impl<T: ParseArg> ParseArg for std::sync::Mutex<T> {
    type Error = T::Error;

    fn parse_arg(arg: &OsStr) -> Result<Self, Self::Error> {
        T::parse_arg(arg).map(Into::into)
    }

    fn describe_type<W: fmt::Write>(writer: W) -> fmt::Result {
        T::describe_type(writer)
    }

    fn parse_owned_arg(arg: OsString) -> Result<Self, Self::Error> {
        T::parse_owned_arg(arg).map(Into::into)
    }
}

impl<T: ParseArg> ParseArg for std::sync::RwLock<T> {
    type Error = T::Error;

    fn parse_arg(arg: &OsStr) -> Result<Self, Self::Error> {
        T::parse_arg(arg).map(Into::into)
    }

    fn describe_type<W: fmt::Write>(writer: W) -> fmt::Result {
        T::describe_type(writer)
    }

    fn parse_owned_arg(arg: OsString) -> Result<Self, Self::Error> {
        T::parse_owned_arg(arg).map(Into::into)
    }
}

impl<T: ParseArg> ParseArg for std::mem::ManuallyDrop<T> {
    type Error = T::Error;

    fn parse_arg(arg: &OsStr) -> Result<Self, Self::Error> {
        T::parse_arg(arg).map(std::mem::ManuallyDrop::new)
    }

    fn describe_type<W: fmt::Write>(writer: W) -> fmt::Result {
        T::describe_type(writer)
    }

    fn parse_owned_arg(arg: OsString) -> Result<Self, Self::Error> {
        T::parse_owned_arg(arg).map(std::mem::ManuallyDrop::new)
    }
}

impl<T: ParseArg> ParseArg for std::rc::Rc<T> {
    type Error = T::Error;

    fn parse_arg(arg: &OsStr) -> Result<Self, Self::Error> {
        T::parse_arg(arg).map(Into::into)
    }

    fn describe_type<W: fmt::Write>(writer: W) -> fmt::Result {
        T::describe_type(writer)
    }

    fn parse_owned_arg(arg: OsString) -> Result<Self, Self::Error> {
        T::parse_owned_arg(arg).map(Into::into)
    }
}

#[cfg(target_has_atomic = "ptr")]
impl<T: ParseArg> ParseArg for std::sync::Arc<T> {
    type Error = T::Error;

    fn parse_arg(arg: &OsStr) -> Result<Self, Self::Error> {
        T::parse_arg(arg).map(Into::into)
    }

    fn describe_type<W: fmt::Write>(writer: W) -> fmt::Result {
        T::describe_type(writer)
    }

    fn parse_owned_arg(arg: OsString) -> Result<Self, Self::Error> {
        T::parse_owned_arg(arg).map(Into::into)
    }
}

/// Possible error when parsing certain arguments.
///
/// This is used for bridging implementations of `FromStr`, because they require UTF-8 encoded inputs.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub enum ParseArgError<E> {
    /// Parsing as implemented by `FromStr` failed.
    FromStr(E),
    /// The input isn't UTF-8 encoded.
    InvalidUtf8,
}

impl<E: fmt::Display> fmt::Display for ParseArgError<E> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            // We flatten the error since we don't have anything to say about it
            ParseArgError::FromStr(err) => fmt::Display::fmt(err, f),
            ParseArgError::InvalidUtf8 => write!(f, "invalid UTF-8 encoding"),
        }
    }
}

/// `Debug` is implemented via `Display` in order to make using `?` operator in `main()` nice.
impl<E: fmt::Display> fmt::Debug for ParseArgError<E> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl<E: fmt::Display> From<E> for ParseArgError<E> {
    fn from(error: E) -> Self {
        ParseArgError::FromStr(error)
    }
}

impl<E: std::error::Error> std::error::Error for ParseArgError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            // We flatten the error since we don't have anything to say about it
            ParseArgError::FromStr(error) => error.source(),
            ParseArgError::InvalidUtf8 => None,
        }
    }
}

/// Shorthand for implementing `ParseArg` for types that already implement `FromStr`.
///
/// This trait allows to implement `ParseArg` cheaply, just by providing the description. Prefer
/// this approach if your type already impls `FromStr` without copying the string. In case the
/// implementation can be made more performant by directly implementing `ParseArg`, prefer direct
/// implementation. (This is what `String` does for example.)
///
/// **This trait should only be implemented - do not use it as a bound! Use `ParseArg` as a bound
/// because it is more general and provides same features.**
pub trait ParseArgFromStr: FromStr where <Self as FromStr>::Err: fmt::Display {
    /// Writes human-readable description of the type to the writer.
    ///
    /// For full information, read the documentation for `ParseArgs::describe_type`.
    ///
    /// `ParseArgs::describe_type` is delegated to this method when `ParseArgFromStr` is
    /// implemented.
    fn describe_type<W: fmt::Write>(writer: W) -> fmt::Result;
}

impl<T> ParseArg for T where T: ParseArgFromStr, <T as FromStr>::Err: fmt::Display {
    type Error = ParseArgError<<T as FromStr>::Err>;

    fn parse_arg(arg: &OsStr) -> Result<Self, Self::Error> {
        arg.to_str().ok_or(ParseArgError::InvalidUtf8)?.parse().map_err(Into::into)
    }

    fn describe_type<W: fmt::Write>(writer: W) -> fmt::Result {
        <Self as ParseArgFromStr>::describe_type(writer)
    }
}

/// Optimized implementation - doesn't allocate in `parse_owned_arg`.
impl ParseArg for String {
    type Error = ParseArgError<std::string::ParseError>;

    fn parse_arg(arg: &OsStr) -> Result<Self, Self::Error> {
        arg.to_str().ok_or(ParseArgError::InvalidUtf8).map(Into::into).map_err(Into::into)
    }

    fn describe_type<W: fmt::Write>(mut writer: W) -> fmt::Result {
        write!(writer, "a UTF-8 encoded string")
    }

    fn parse_owned_arg(arg: OsString) -> Result<Self, Self::Error> {
        arg.into_string().map_err(|_| ParseArgError::InvalidUtf8)
    }
}

/// This implementation is a no-op or clone, since `OsString` is already `OsString`.
impl ParseArg for OsString {
    type Error = std::string::ParseError;

    fn parse_arg(arg: &OsStr) -> Result<Self, Self::Error> {
        Ok(arg.into())
    }

    fn describe_type<W: fmt::Write>(mut writer: W) -> fmt::Result {
        write!(writer, "any string")
    }

    fn parse_owned_arg(arg: OsString) -> Result<Self, Self::Error> {
        Ok(arg)
    }
}

/// To my knowledge this is a no-op or clone.
impl ParseArg for std::path::PathBuf {
    type Error = std::string::ParseError;

    fn parse_arg(arg: &OsStr) -> Result<Self, Self::Error> {
        Ok(arg.into())
    }

    fn describe_type<W: fmt::Write>(mut writer: W) -> fmt::Result {
        write!(writer, "a path")
    }

    fn parse_owned_arg(arg: OsString) -> Result<Self, Self::Error> {
        Ok(arg.into())
    }
}

macro_rules! impl_unsized {
    ($($type:ty),*) => {
        $(
            /// This implementation simply parses the "fat" owned type and converts it into the box.
            impl ParseArg for Box<$type> {
                type Error = <<$type as ToOwned>::Owned as ParseArg>::Error;

                fn parse_arg(arg: &OsStr) -> Result<Self, Self::Error> {
                    <<$type as ToOwned>::Owned as ParseArg>::parse_arg(arg).map(Into::into)
                }

                fn describe_type<W: fmt::Write>(writer: W) -> fmt::Result {
                    <<$type as ToOwned>::Owned as ParseArg>::describe_type(writer)
                }

                fn parse_owned_arg(arg: OsString) -> Result<Self, Self::Error> {
                    <<$type as ToOwned>::Owned as ParseArg>::parse_owned_arg(arg).map(Into::into)
                }
            }

            impl ParseArg for std::borrow::Cow<'static, $type> {
                type Error = <<$type as ToOwned>::Owned as ParseArg>::Error;

                fn parse_arg(arg: &OsStr) -> Result<Self, Self::Error> {
                    <<$type as ToOwned>::Owned as ParseArg>::parse_arg(arg).map(Into::into)
                }

                fn describe_type<W: fmt::Write>(writer: W) -> fmt::Result {
                    <<$type as ToOwned>::Owned as ParseArg>::describe_type(writer)
                }

                fn parse_owned_arg(arg: OsString) -> Result<Self, Self::Error> {
                    <<$type as ToOwned>::Owned as ParseArg>::parse_owned_arg(arg).map(Into::into)
                }
            }
        )*
    }
}

impl_unsized!(str, std::path::Path, std::ffi::OsStr);

macro_rules! impl_trivial_rc {
    ($($type:ty),*) => {
        $(
            impl ParseArg for std::rc::Rc<$type> {
                type Error = std::convert::Infallible;

                fn parse_arg(arg: &OsStr) -> Result<Self, Self::Error> {
                    let arg: &$type = arg.as_ref();
                    Ok(arg.into())
                }

                fn describe_type<W: fmt::Write>(writer: W) -> fmt::Result {
                    <<$type as ToOwned>::Owned as ParseArg>::describe_type(writer)
                }
            }

            #[cfg(target_has_atomic = "ptr")]
            impl ParseArg for std::sync::Arc<$type> {
                type Error = std::convert::Infallible;

                fn parse_arg(arg: &OsStr) -> Result<Self, Self::Error> {
                    let arg: &$type = arg.as_ref();
                    Ok(arg.into())
                }

                fn describe_type<W: fmt::Write>(writer: W) -> fmt::Result {
                    <<$type as ToOwned>::Owned as ParseArg>::describe_type(writer)
                }
            }
        )*
    }
}

impl_trivial_rc!(std::path::Path, std::ffi::OsStr);

impl ParseArg for std::rc::Rc<str> {
    type Error = ParseArgError<std::string::ParseError>;

    fn parse_arg(arg: &OsStr) -> Result<Self, Self::Error> {
        arg.to_str().map(Into::into).ok_or(ParseArgError::InvalidUtf8)
    }

    fn describe_type<W: fmt::Write>(writer: W) -> fmt::Result {
        String::describe_type(writer)
    }
}

#[cfg(target_has_atomic = "ptr")]
impl ParseArg for std::sync::Arc<str> {
    type Error = ParseArgError<std::string::ParseError>;

    fn parse_arg(arg: &OsStr) -> Result<Self, Self::Error> {
        arg.to_str().map(Into::into).ok_or(ParseArgError::InvalidUtf8)
    }

    fn describe_type<W: fmt::Write>(writer: W) -> fmt::Result {
        String::describe_type(writer)
    }
}

macro_rules! impl_unsigned {
    ($($type:ty),*) => {
        $(
            impl ParseArgFromStr for $type {
                fn describe_type<W: fmt::Write>(mut writer: W) -> fmt::Result {
                    write!(writer, "a non-negative integer up to {}", <$type>::MAX)
                }
            }
        )*
    }
}

macro_rules! impl_signed {
    ($($type:ty),*) => {
        $(
            impl ParseArgFromStr for $type {
                fn describe_type<W: fmt::Write>(mut writer: W) -> fmt::Result {
                    write!(writer, "an integer at least {} and up to {}", <$type>::MIN, <$type>::MAX)
                }
            }
        )*
    }
}

const fn max_from_len(len: usize) -> u128 {
    assert!(len <= 16);

    let mut bytes = [0; 16];
    let mut i = 0;
    while i < len {
        bytes[i] = 0xFF;
        i += 1;
    }
    u128::from_le_bytes(bytes)
}

const fn max_from_len_signed(len: usize) -> i128 {
    (max_from_len(len) / 2) as i128
}

const fn min_from_len_signed(len: usize) -> i128 {
    -((max_from_len(len) / 2) as i128) - 1
}

macro_rules! impl_non_zero_unsigned {
    ($($type:ident),*) => {
        $(
            impl ParseArgFromStr for std::num::$type {
                fn describe_type<W: fmt::Write>(mut writer: W) -> fmt::Result {
                    const MAX: u128 = max_from_len(std::mem::size_of::<std::num::$type>());
                    write!(writer, "a positive (non-zero) integer up to {}", MAX)
                }
            }
        )*
    }
}

macro_rules! impl_non_zero_signed {
    ($($type:ident),*) => {
        $(
            impl ParseArgFromStr for std::num::$type {
                fn describe_type<W: fmt::Write>(mut writer: W) -> fmt::Result {
                    const MAX: i128 = max_from_len_signed(std::mem::size_of::<std::num::$type>());
                    const MIN: i128 = min_from_len_signed(std::mem::size_of::<std::num::$type>());
                    write!(writer, "a non-zero integer at least {} and up to {}", MIN, MAX)
                }
            }
        )*
    }
}

fn parse_and_convert<Intermediate: FromStr, Res, F: FnOnce(Intermediate) -> Res>(s: &str, f: F) -> Result<Res, Intermediate::Err> {
    s.parse().map(f)
}

macro_rules! impl_atomic_unsigned {
    ($($type:ident => $atomic_kind:literal),*) => {
        $(
            #[cfg(target_has_atomic = $atomic_kind)]
            impl ParseArg for std::sync::atomic::$type {
                type Error = ParseArgError<std::num::ParseIntError>;

                fn parse_arg(arg: &OsStr) -> Result<Self, Self::Error> {
                    parse_and_convert(arg.to_str().ok_or(ParseArgError::InvalidUtf8)?, std::sync::atomic::$type::new).map_err(Into::into)
                }

                fn describe_type<W: fmt::Write>(mut writer: W) -> fmt::Result {
                    const MAX: u128 = max_from_len(std::mem::size_of::<std::sync::atomic::$type>());
                    write!(writer, "a non-negative integer up to {}", MAX)
                }
            }
        )*
    }
}

macro_rules! impl_atomic_signed {
    ($($type:ident => $atomic_kind:literal),*) => {
        $(
            #[cfg(target_has_atomic = $atomic_kind)]
            impl ParseArg for std::sync::atomic::$type {
                type Error = ParseArgError<std::num::ParseIntError>;

                fn parse_arg(arg: &OsStr) -> Result<Self, Self::Error> {
                    parse_and_convert(arg.to_str().ok_or(ParseArgError::InvalidUtf8)?, std::sync::atomic::$type::new).map_err(Into::into)
                }

                fn describe_type<W: fmt::Write>(mut writer: W) -> fmt::Result {
                    const MAX: i128 = max_from_len_signed(std::mem::size_of::<std::sync::atomic::$type>());
                    const MIN: i128 = min_from_len_signed(std::mem::size_of::<std::sync::atomic::$type>());
                    write!(writer, "an integer at least {} and up to {}", MIN, MAX)
                }
            }
        )*
    }
}

macro_rules! impl_float {
    ($($type:ident),*) => {
        $(
            impl ParseArgFromStr for $type {
                fn describe_type<W: fmt::Write>(mut writer: W) -> fmt::Result {
                    write!(writer, "a real number at least {} and up to {}", std::$type::MIN, std::$type::MAX)
                }
            }
        )*
    }
}

impl_unsigned! { u8, u16, u32, u64, u128, usize }
impl_signed! { i8, i16, i32, i64, i128, isize }
impl_non_zero_unsigned! { NonZeroU8, NonZeroU16, NonZeroU32, NonZeroU64, NonZeroU128, NonZeroUsize }
impl_non_zero_signed! { NonZeroI8, NonZeroI16, NonZeroI32, NonZeroI64, NonZeroI128, NonZeroIsize }
impl_atomic_unsigned! { AtomicU8 => "8", AtomicU16 => "16", AtomicU32 => "32", AtomicU64 => "64", AtomicU128 => "128", AtomicUsize => "ptr" }
impl_atomic_signed! { AtomicI8 => "8", AtomicI16 => "16", AtomicI32 => "32", AtomicI64 => "64", AtomicI128 => "128", AtomicIsize => "ptr" }
impl_float! { f32, f64 }

impl ParseArgFromStr for std::net::IpAddr {
    fn describe_type<W: fmt::Write>(mut writer: W) -> fmt::Result {
        write!(writer, "an IP address (either version 4 or 6)")
    }
}

impl ParseArgFromStr for std::net::Ipv4Addr {
    fn describe_type<W: fmt::Write>(mut writer: W) -> fmt::Result {
        write!(writer, "a version 4 IP address")
    }
}

impl ParseArgFromStr for std::net::Ipv6Addr {
    fn describe_type<W: fmt::Write>(mut writer: W) -> fmt::Result {
        write!(writer, "a version 6 IP address")
    }
}

impl ParseArgFromStr for std::net::SocketAddr {
    fn describe_type<W: fmt::Write>(mut writer: W) -> fmt::Result {
        write!(writer, "a version 4 or 6 network socket address (IP:port)")
    }
}

impl ParseArgFromStr for std::net::SocketAddrV4 {
    fn describe_type<W: fmt::Write>(mut writer: W) -> fmt::Result {
        write!(writer, "a version 4 socket address (IP:port)")
    }
}

impl ParseArgFromStr for std::net::SocketAddrV6 {
    fn describe_type<W: fmt::Write>(mut writer: W) -> fmt::Result {
        write!(writer, "a version 6 socket address (IP:port)")
    }
}

impl ParseArgFromStr for bool {
    fn describe_type<W: fmt::Write>(mut writer: W) -> fmt::Result {
        write!(writer, "a boolean (true or false)")
    }
}

impl ParseArgFromStr for char {
    fn describe_type<W: fmt::Write>(mut writer: W) -> fmt::Result {
        write!(writer, "a single character (Unicode code point)")
    }
}

/// Error that can occur durin parsing command-line argument.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ValueError<E> {
    /// The value couldn't be parsed because of the invalid format.
    InvalidValue(E),
    /// The value to the argument is missing - too few arguments.
    MissingValue,
}

impl<E> ValueError<E> {
    /// Applies function to the contained error (if any), or returns the provided default (if not).
    ///
    /// This is practically the same thing what `Option<E>` would do.
    pub fn map_or<R, F: FnOnce(E) -> R>(self, default: R, f: F) -> R {
        match self {
            ValueError::InvalidValue(error) => f(error),
            ValueError::MissingValue => default,
        }
    }
}

/// A type that can be parsed.
///
/// Usually `&OsStr` or `OsString`. It's used to automatically pick the right
/// method of `ParseArg` trait.
pub trait Arg: Sized + AsRef<OsStr> {
    /// Parses the argument from `self` using appropriate method.
    ///
    /// In case `self` is `OsString` or can be converted to it at no cost
    /// `ParseArg::parse_owned_arg()` is used. Otherwise `ParseArg::parse_arg()` is used.
    fn parse<T: ParseArg>(self) -> Result<T, <T as ParseArg>::Error>;
}

impl Arg for OsString {
    fn parse<T: ParseArg>(self) -> Result<T, <T as ParseArg>::Error> {
        T::parse_owned_arg(self)
    }
}

impl<'a> Arg for &'a OsStr {
    fn parse<T: ParseArg>(self) -> Result<T, <T as ParseArg>::Error> {
        T::parse_arg(self)
    }
}

impl<'a, U: 'a + Arg + Copy> Arg for &'a U {
    fn parse<T: ParseArg>(self) -> Result<T, <T as ParseArg>::Error> {
        U::parse(*self)
    }
}

/// Using this with `std::env::args()` is not a good practice, but it may be
/// useful for testing.
impl Arg for String {
    fn parse<T: ParseArg>(self) -> Result<T, <T as ParseArg>::Error> {
        T::parse_owned_arg(self.into())
    }
}

/// Using this with `std::env::args()` is not a good practice, but it may be
/// useful for testing.
impl<'a> Arg for &'a str {
    fn parse<T: ParseArg>(self) -> Result<T, <T as ParseArg>::Error> {
        T::parse_arg(self.as_ref())
    }
}

/// Checks whether given arg matches the `name` or begins with `name=` and parses it
/// appropriately.
///
/// In case `arg` is equal to name, the next argument is pulled from `next` iterator and parsed using
/// the most efficient method.
/// In case `arg` begins with `name=`, it strips the `name=` prefix and parses the rest.
///
/// Note: due to a limitation of `std`, Windows implementation is slower than Unix one in `name=`
/// case. The difference is one-two allocations (depending on parsed type). This probably won't be
/// noticable for the user.
pub fn match_arg<T: ParseArg, S: AsRef<OsStr>, I>(name: &str, arg: S, next: I) -> Option<Result<T, ValueError<<T as ParseArg>::Error>>> where I: IntoIterator, I::Item: Arg {
    if *(arg.as_ref()) == *name {
        if let Some(arg) = next.into_iter().next() {
            Some(arg.parse().map_err(ValueError::InvalidValue))
        } else {
            Some(Err(ValueError::MissingValue))
        }
    } else {
        check_prefix::<T>(name.as_ref(), arg.as_ref()).map(|result| result.map_err(ValueError::InvalidValue))
    }
}

/// Creates an iterator of short arguments if the input is in the form `-abc`.
///
/// This is a helper for parsing short arguments in the same way many GNU commands support.
/// It enables putting several flags into a single argument and enables the last one to be
/// a short parameter with a value and the value might be either concatenated or put as the
/// next argument.
///
/// In order to use this, one would call this function and, if it returned `Some(iter)`, then
/// iterate `iter` and match returned chars. If a char representing a parameter with value
/// is returned from `next`, `.parse_remaining()` method should be called on the iterator.
/// It will attempt to parse the value from remaining part, if present. If there's none, it'll
/// attempt to parse it from the following argument pulled from the parameter.
///
/// This is how `configure_me` uses this feature, but you can use it without `configure_me`.
#[inline]
pub fn iter_short<'a, T>(arg: &'a T) -> Option<ShortIter<'a>> where T: AsRef<OsStr> + ?Sized {
    iter_short_internal(arg.as_ref())
}

#[cfg(unix)]
fn iter_short_internal<'a>(arg: &'a OsStr) -> Option<ShortIter<'a>> {
    use ::std::os::unix::ffi::OsStrExt;

    let slice = arg.as_bytes();
    // Needed to avoid panic and this can't be an option anyway
    if slice.len() < 2 {
        return None;
    }

    // Check for things beginning with --
    if slice[1] == b'-' {
        return None;
    }

    let mut arg_iter = slice.iter();
    if *arg_iter.next()? == b'-' {
        Some(ShortIter {
            iter: arg_iter,
        })
    } else {
        None
    }
}

#[cfg(windows)]
fn iter_short_internal<'a>(arg: &'a OsStr) -> Option<ShortIter<'a>> {
    use ::std::os::windows::ffi::OsStrExt;

    let mut iter = arg.encode_wide();

    if iter.next()? == u16::from(b'-') {
        let mut iter = iter.peekable();
        if *iter.peek()? != u16::from(b'-') {
            Some(ShortIter {
                iter,
            })
        } else {
            None
        }
    } else {
        None
    }
}

/// An iterator of short options.
///
/// See the documentation for `iter_short` for more details.
// In case the platform is neither unix nor windows, we want to keep this private
#[non_exhaustive]
pub struct ShortIter<'a> {
    #[cfg(unix)]
    iter: std::slice::Iter<'a, u8>,
    #[cfg(windows)]
    iter: std::iter::Peekable<std::os::windows::ffi::EncodeWide<'a>>,
}

impl<'a> Iterator for ShortIter<'a> {
    type Item = char;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_impl()
    }
}

impl<'a> ShortIter<'a> {
    /// Parses the remaining string of the argument.
    ///
    /// This is used in case of `-xVAL` style arguments. If `VAL` is missing, it is pulled from
    /// `iter`. If that's missing too, Err(ValueError::MissingValue) is returned.
    ///
    /// See the doc of `iter_short()` for more context.
    ///
    /// While this method could take `&self`, consuming expresses that the iteration should end.
    pub fn parse_remaining<T: ParseArg, I>(self, iter: I) -> Result<T, ValueError<T::Error>> where I: IntoIterator, I::Item: Arg {
        self.parse_remaining_internal(iter)
    }

    #[cfg(unix)]
    fn parse_remaining_internal<T: ParseArg, I>(self, iter: I) -> Result<T, ValueError<T::Error>> where I: IntoIterator, I::Item: Arg {
        use ::std::os::unix::ffi::OsStrExt;

        let slice = self.iter.as_slice();
        if slice.len() == 0 {
            return iter
                .into_iter()
                .next()
                .map_or(Err(ValueError::MissingValue), |val| val.parse().map_err(ValueError::InvalidValue));
        }

        OsStr::from_bytes(slice)
            .parse()
            .map_err(ValueError::InvalidValue)
    }

    #[cfg(windows)]
    fn parse_remaining_internal<T: ParseArg, I>(mut self, iter: I) -> Result<T, ValueError<T::Error>> where I: IntoIterator, I::Item: Arg {
        use ::std::os::windows::ffi::OsStringExt;

        if self.iter.peek().is_none() {
            return iter
                .into_iter()
                .next()
                .map_or(Err(ValueError::MissingValue), |val| val.parse().map_err(ValueError::InvalidValue));
        }

        let arg = self.iter.collect::<Vec<_>>();
        let arg = OsString::from_wide(&arg);
        arg
            .parse()
            .map_err(ValueError::InvalidValue)
    }

    #[cfg(unix)]
    fn next_impl(&mut self) -> Option<<Self as Iterator>::Item> {
        self.iter.next().map(|&c| c.into())
    }

    #[cfg(windows)]
    fn next_impl(&mut self) -> Option<<Self as Iterator>::Item> {
        self.iter.next().and_then(|c| {
            // We assume all options are ASCII chars, which should be decoded as exactly one char.
            // If they're not, we stop iteration.
            //
            // TODO: might be better to return error instead.
            let mut decoder = std::char::decode_utf16(std::iter::once(c));
            let result = decoder.next()?.ok()?;
            if decoder.next().is_none() {
                Some(result)
            } else {
                None
            }
        })
    }
}

#[cfg(unix)]
fn check_prefix<T: ParseArg>(name: &OsStr, arg: &OsStr) -> Option<Result<T, <T as ParseArg>::Error>> {
    use ::std::os::unix::ffi::OsStrExt;

    let mut arg_iter = arg.as_bytes().iter();
    if name.as_bytes().iter().zip(&mut arg_iter).all(|(a, b)| *a == *b) {
        if arg_iter.next() == Some(&b'=') {
            let arg = OsStr::from_bytes(arg_iter.as_slice());
            Some(T::parse_arg(arg))
        } else {
            None
        }
    } else {
        None
    }
}

#[cfg(windows)]
fn check_prefix<T: ParseArg>(name: &OsStr, arg: &OsStr) -> Option<Result<T, <T as ParseArg>::Error>> {
    use ::std::os::windows::ffi::{OsStrExt, OsStringExt};

    let mut arg_iter = arg.encode_wide();
    if name.encode_wide().zip(&mut arg_iter).all(|(a, b)| a == b) {
        if arg_iter.next() == Some(u16::from(b'=')) {
            // If you don't like these two allocations, go annoy Rust libs team
            // with RFC to implement matching and slicing on `OsStr`...
            let arg = arg_iter.collect::<Vec<_>>();
            let arg = OsString::from_wide(&arg);
            Some(T::parse_owned_arg(arg))
        } else {
            None
        }
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn numbers() {
        use ::ParseArg;

        let val: u8 = ParseArg::parse_arg("42".as_ref()).unwrap();
        assert_eq!(val, 42);
        let val: u16 = ParseArg::parse_arg("42".as_ref()).unwrap();
        assert_eq!(val, 42);
        let val: u32 = ParseArg::parse_arg("42".as_ref()).unwrap();
        assert_eq!(val, 42);
        let val: u64 = ParseArg::parse_arg("42".as_ref()).unwrap();
        assert_eq!(val, 42);
        let val: u128 = ParseArg::parse_arg("42".as_ref()).unwrap();
        assert_eq!(val, 42);

        let val: usize = ParseArg::parse_arg("42".as_ref()).unwrap();
        assert_eq!(val, 42);
        let val: i8 = ParseArg::parse_arg("42".as_ref()).unwrap();
        assert_eq!(val, 42);
        let val: i16 = ParseArg::parse_arg("42".as_ref()).unwrap();
        assert_eq!(val, 42);
        let val: i32 = ParseArg::parse_arg("42".as_ref()).unwrap();
        assert_eq!(val, 42);
        let val: i64 = ParseArg::parse_arg("42".as_ref()).unwrap();
        assert_eq!(val, 42);
        let val: i128 = ParseArg::parse_arg("42".as_ref()).unwrap();
        assert_eq!(val, 42);

        let val: isize = ParseArg::parse_arg("-42".as_ref()).unwrap();
        assert_eq!(val, -42);
        let val: i8 = ParseArg::parse_arg("-42".as_ref()).unwrap();
        assert_eq!(val, -42);
        let val: i16 = ParseArg::parse_arg("-42".as_ref()).unwrap();
        assert_eq!(val, -42);
        let val: i32 = ParseArg::parse_arg("-42".as_ref()).unwrap();
        assert_eq!(val, -42);
        let val: i64 = ParseArg::parse_arg("-42".as_ref()).unwrap();
        assert_eq!(val, -42);
        let val: i128 = ParseArg::parse_arg("-42".as_ref()).unwrap();
        assert_eq!(val, -42);
        let val: isize = ParseArg::parse_arg("42".as_ref()).unwrap();
        assert_eq!(val, 42);

        let val: f32 = ParseArg::parse_arg("42.42".as_ref()).unwrap();
        assert_eq!(val, 42.42);
        let val: f64 = ParseArg::parse_arg("42.42".as_ref()).unwrap();
        assert_eq!(val, 42.42);
    }

    #[test]
    fn match_args() {
        assert_eq!(::match_arg::<u32, _, _>("--foo", "--bar", std::iter::empty::<&str>()), None);
        assert_eq!(::match_arg::<u32, _, _>("--foo", "--bar", &["--foo"]), None);
        assert_eq!(::match_arg::<u32, _, _>("--foo", "--foo", std::iter::empty::<&str>()), Some(Err(::ValueError::MissingValue)));
        assert_eq!(::match_arg::<u32, _, _>("--foo", "--foo", &["--foo"]), Some("--foo".parse::<u32>().map_err(::ParseArgError::FromStr).map_err(::ValueError::InvalidValue)));
        assert_eq!(::match_arg::<u32, _, _>("--foo", "--foo", &["42"]), Some(Ok(42)));


        assert_eq!(::match_arg::<u32, _, _>("--foo", "--bar=", std::iter::empty::<&str>()), None);
        assert_eq!(::match_arg::<u32, _, _>("--foo", "--bar=", &["--foo"]), None);
        assert_eq!(::match_arg::<u32, _, _>("--foo", "--foo=", std::iter::empty::<&str>()), Some("".parse::<u32>().map_err(::ParseArgError::FromStr).map_err(::ValueError::InvalidValue)));
        let mut iter = ["--foo"].iter();
        assert_eq!(::match_arg::<u32, _, _>("--foo", "--foo=--foo", &mut iter), Some("--foo".parse::<u32>().map_err(::ParseArgError::FromStr).map_err(::ValueError::InvalidValue)));
        let mut iter = ["47"].iter();
        assert_eq!(::match_arg::<u32, _, _>("--foo", "--foo=42", &mut iter), Some(Ok(42)));
        assert_eq!(iter.next(), Some(&"47"));
    }

    #[test]
    fn iter_short() {
        use ::ValueError;

        assert!(::iter_short("").is_none());
        assert!(::iter_short("-").is_none());
        assert!(::iter_short("--").is_none());
        assert!(::iter_short("--a").is_none());
        assert!(::iter_short("--ab").is_none());
        assert_eq!(::iter_short("-a").expect("Iter").next().expect("next"), 'a');
        let mut iter = ::iter_short("-ab").expect("Iter");
        assert_eq!(iter.next().expect("next"), 'a');
        assert_eq!(iter.next().expect("next"), 'b');
        assert!(iter.next().is_none());

        let mut iter = ::iter_short("-abcde").expect("Iter");
        assert_eq!(iter.next().expect("next"), 'a');
        assert_eq!(iter.next().expect("next"), 'b');
        assert_eq!(iter.next().expect("next"), 'c');
        assert_eq!(iter.next().expect("next"), 'd');
        assert_eq!(iter.next().expect("next"), 'e');
        assert!(iter.next().is_none());

        let mut iter = ::iter_short("-a").expect("Iter");
        assert_eq!(iter.next().expect("next"), 'a');
        assert_eq!(iter.parse_remaining::<u32, _>(::std::iter::empty::<&str>()), Err(ValueError::MissingValue));

        let mut iter = ::iter_short("-a").expect("Iter");
        assert_eq!(iter.next().expect("next"), 'a');
        assert_eq!(iter.parse_remaining::<u32, _>(&["42"]).expect("Failed to parse"), 42);

        let mut iter = ::iter_short("-a42").expect("Iter");
        assert_eq!(iter.next().expect("next"), 'a');
        assert_eq!(iter.parse_remaining::<u32, _>(::std::iter::empty::<&str>()).expect("Failed to parse"), 42);

        let mut iter = ::iter_short("-a42").expect("Iter");
        assert_eq!(iter.next().expect("next"), 'a');
        assert_eq!(iter.parse_remaining::<u32, _>(&["24"]).expect("Failed to parse"), 42);

        let mut iter = ::iter_short("-ab42").expect("Iter");
        assert_eq!(iter.next().expect("next"), 'a');
        assert_eq!(iter.next().expect("next"), 'b');
        assert_eq!(iter.parse_remaining::<u32, _>(::std::iter::empty::<&str>()).expect("Failed to parse"), 42);

        let mut iter = ::iter_short("-ab42").expect("Iter");
        assert_eq!(iter.next().expect("next"), 'a');
        assert_eq!(iter.next().expect("next"), 'b');
        assert_eq!(iter.parse_remaining::<u32, _>(&["24"]).expect("Failed to parse"), 42);

        let mut iter = ::iter_short("-abc").expect("Iter");
        assert_eq!(iter.next().expect("next"), 'a');
        assert_eq!(iter.next().expect("next"), 'b');
        match iter.parse_remaining::<u32, _>(&["24"]) {
            Ok(val) => panic!("Parsed unexpected vlaue: {}, parsing should've failed", val),
            Err(ValueError::MissingValue) => panic!("Value shouldn't be missing"),
            Err(ValueError::InvalidValue(_)) => (),
        }
    }

    #[test]
    fn min_max() {
        fn check<T: Into<u128>>(max: T) {
            let max = max.into();
            assert_eq!(super::max_from_len(std::mem::size_of::<T>()), max);
        }

        fn check_signed<T: Into<i128>>(min: T, max: T) {
            let min = min.into();
            let max = max.into();
            assert_eq!(super::min_from_len_signed(std::mem::size_of::<T>()), min);
            assert_eq!(super::max_from_len_signed(std::mem::size_of::<T>()), max);
        }

        check(u8::MAX);
        check(u16::MAX);
        check(u128::MAX);
        check_signed(i8::MIN, i8::MAX);
        check_signed(i16::MIN, i16::MAX);
        check_signed(i128::MIN, i128::MAX);
    }
}
