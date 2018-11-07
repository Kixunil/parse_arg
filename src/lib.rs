//! This crate provides traits and implementations for parsing command-line arguments.
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

#![deny(missing_docs)]

use std::fmt;
use std::str::FromStr;
use std::ffi::{OsStr, OsString};

/// Defines an interface for types that can be created by parsing command-line argument.
///
/// This trait is similar to `FromStr`. See the crate documentation for list of importatn
/// differences.
pub trait ParseArg: Sized {
    /// Type returned in `Err` variant of `Result` when parsing fails.
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
            ParseArgError::FromStr(err) => fmt::Display::fmt(err, f),
            ParseArgError::InvalidUtf8 => write!(f, "invalid UTF-8 encoding"),
        }
    }
}

/// `Debug` is implemented via `Display` in order to make using `?` operator in main nice.
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
    /// implmented.
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

macro_rules! impl_unsigned {
    ($($type:ty),*) => {
        $(
            impl ParseArgFromStr for $type {
                fn describe_type<W: fmt::Write>(mut writer: W) -> fmt::Result {
                    write!(writer, "a non-negative integer up to {}", <$type>::max_value())
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
                    write!(writer, "an integer at least {} and up to {}", <$type>::min_value(), <$type>::max_value())
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
impl_float! { f32, f64 }

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

/// Error that can occur durin parsing command-line argument.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ValueError<E> {
    /// The value couldn't be parsed because of the invalid format.
    InvalidValue(E),
    /// The value to the argument is missing - too few arguments.
    MissingValue,
}

/// A type that can be parsed.
///
/// Usually `&OsStr` or `OsString`. It's used to automatically pick the right
/// method if `ParseArg` trait.
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
}
