//! PTX Lexer - Tokenization of PTX source code

use super::error::ParseError;
use super::ast::SourceLocation;

/// Token kinds for PTX lexing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    /// End of file
    Eof,
    /// `.version`, `.target`, `.address_size`, etc.
    Directive,
    /// .entry keyword
    Entry,
    /// .func keyword
    Func,
    /// .reg declaration
    Reg,
    /// .shared declaration
    Shared,
    /// .local declaration
    Local,
    /// .global declaration
    Global,
    /// .param declaration
    Param,
    /// Identifier (register name, label, etc.)
    Identifier,
    /// Integer literal
    Integer,
    /// Float literal
    Float,
    /// Instruction (ld, st, mov, add, etc.)
    Instruction,
    /// Label (name:)
    Label,
    /// Comment (// or /* */)
    Comment,
    /// Opening brace {
    LBrace,
    /// Closing brace }
    RBrace,
    /// Opening parenthesis (
    LParen,
    /// Closing parenthesis )
    RParen,
    /// Opening bracket [
    LBracket,
    /// Closing bracket ]
    RBracket,
    /// Comma ,
    Comma,
    /// Semicolon ;
    Semicolon,
    /// Colon :
    Colon,
    /// Unknown token
    Unknown,
}

/// A token in the PTX source
#[derive(Debug, Clone)]
pub struct Token {
    /// The kind of token
    pub kind: TokenKind,
    /// The text of the token
    pub text: String,
    /// Source location
    pub location: SourceLocation,
}

impl Default for Token {
    fn default() -> Self {
        Self {
            kind: TokenKind::Eof,
            text: String::new(),
            location: SourceLocation::default(),
        }
    }
}

/// PTX Lexer
pub struct Lexer<'a> {
    source: &'a str,
    pos: usize,
    line: usize,
    column: usize,
}

impl<'a> Lexer<'a> {
    /// Create a new lexer for the given source
    pub fn new(source: &'a str) -> Self {
        Self {
            source,
            pos: 0,
            line: 1,
            column: 1,
        }
    }

    fn peek(&self) -> Option<char> {
        self.source[self.pos..].chars().next()
    }

    fn peek_at(&self, offset: usize) -> Option<char> {
        self.source[self.pos..].chars().nth(offset)
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        if c == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(c)
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.advance();
            } else if c == '/' {
                if self.peek_at(1) == Some('/') {
                    // Line comment
                    while let Some(c) = self.peek() {
                        self.advance();
                        if c == '\n' {
                            break;
                        }
                    }
                } else if self.peek_at(1) == Some('*') {
                    // Block comment
                    self.advance(); // skip /
                    self.advance(); // skip *
                    while let Some(c) = self.peek() {
                        self.advance();
                        if c == '*' && self.peek() == Some('/') {
                            self.advance();
                            break;
                        }
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    /// Get the next token from the source
    pub fn next_token(&mut self) -> Result<Token, ParseError> {
        self.skip_whitespace();

        let location = SourceLocation {
            line: self.line,
            column: self.column,
            file: None,
        };

        let Some(c) = self.peek() else {
            return Ok(Token {
                kind: TokenKind::Eof,
                text: String::new(),
                location,
            });
        };

        match c {
            '{' => { self.advance(); Ok(Token { kind: TokenKind::LBrace, text: "{".into(), location }) }
            '}' => { self.advance(); Ok(Token { kind: TokenKind::RBrace, text: "}".into(), location }) }
            '(' => { self.advance(); Ok(Token { kind: TokenKind::LParen, text: "(".into(), location }) }
            ')' => { self.advance(); Ok(Token { kind: TokenKind::RParen, text: ")".into(), location }) }
            '[' => { self.advance(); Ok(Token { kind: TokenKind::LBracket, text: "[".into(), location }) }
            ']' => { self.advance(); Ok(Token { kind: TokenKind::RBracket, text: "]".into(), location }) }
            ',' => { self.advance(); Ok(Token { kind: TokenKind::Comma, text: ",".into(), location }) }
            ';' => { self.advance(); Ok(Token { kind: TokenKind::Semicolon, text: ";".into(), location }) }
            '.' => self.read_directive(location),
            '%' => self.read_register(location),
            '@' => self.read_predicate(location),
            '0'..='9' | '-' => self.read_number(location),
            _ if c.is_alphabetic() || c == '_' => self.read_identifier_or_instruction(location),
            _ => {
                self.advance();
                Ok(Token { kind: TokenKind::Unknown, text: c.to_string(), location })
            }
        }
    }

    fn read_directive(&mut self, location: SourceLocation) -> Result<Token, ParseError> {
        let start = self.pos;
        self.advance(); // skip '.'

        // Read directive name
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }

        let directive_name = &self.source[start..self.pos];

        // For version, target, address_size - read the value too
        let text = if directive_name.starts_with(".version")
            || directive_name.starts_with(".target")
            || directive_name.starts_with(".address_size")
        {
            self.skip_whitespace();
            let value_start = self.pos;
            while let Some(c) = self.peek() {
                if c == '\n' || c == ';' || c == '{' || c == '(' {
                    break;
                }
                self.advance();
            }
            format!("{} {}", directive_name, self.source[value_start..self.pos].trim())
        } else {
            directive_name.to_string()
        };

        let kind = self.classify_directive(&text);
        Ok(Token { kind, text, location })
    }

    fn classify_directive(&self, text: &str) -> TokenKind {
        if text.starts_with(".entry") {
            TokenKind::Entry
        } else if text.starts_with(".func") {
            TokenKind::Func
        } else if text.starts_with(".reg") {
            TokenKind::Reg
        } else if text.starts_with(".shared") {
            TokenKind::Shared
        } else if text.starts_with(".local") {
            TokenKind::Local
        } else if text.starts_with(".global") {
            TokenKind::Global
        } else if text.starts_with(".param") {
            TokenKind::Param
        } else {
            TokenKind::Directive
        }
    }

    fn read_register(&mut self, location: SourceLocation) -> Result<Token, ParseError> {
        let start = self.pos;
        self.advance(); // skip '%'

        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }

        Ok(Token {
            kind: TokenKind::Identifier,
            text: self.source[start..self.pos].to_string(),
            location,
        })
    }

    fn read_predicate(&mut self, location: SourceLocation) -> Result<Token, ParseError> {
        let start = self.pos;
        self.advance(); // skip '@'

        // May have '!' for negation
        if self.peek() == Some('!') {
            self.advance();
        }

        // Read predicate register name
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' || c == '%' {
                self.advance();
            } else {
                break;
            }
        }

        Ok(Token {
            kind: TokenKind::Identifier,
            text: self.source[start..self.pos].to_string(),
            location,
        })
    }

    fn read_number(&mut self, location: SourceLocation) -> Result<Token, ParseError> {
        let start = self.pos;

        // Handle negative sign
        if self.peek() == Some('-') {
            self.advance();
        }

        // Handle hex prefix
        if self.peek() == Some('0') {
            self.advance();
            if matches!(self.peek(), Some('x' | 'X')) {
                self.advance();
                while let Some(c) = self.peek() {
                    if c.is_ascii_hexdigit() {
                        self.advance();
                    } else {
                        break;
                    }
                }
                return Ok(Token {
                    kind: TokenKind::Integer,
                    text: self.source[start..self.pos].to_string(),
                    location,
                });
            }
        }

        // Read integer part
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                self.advance();
            } else {
                break;
            }
        }

        // Check for float
        let mut is_float = false;
        if self.peek() == Some('.') {
            is_float = true;
            self.advance();
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    self.advance();
                } else {
                    break;
                }
            }
        }

        // Check for exponent
        if matches!(self.peek(), Some('e' | 'E')) {
            is_float = true;
            self.advance();
            if matches!(self.peek(), Some('+' | '-')) {
                self.advance();
            }
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    self.advance();
                } else {
                    break;
                }
            }
        }

        Ok(Token {
            kind: if is_float { TokenKind::Float } else { TokenKind::Integer },
            text: self.source[start..self.pos].to_string(),
            location,
        })
    }

    fn read_identifier_or_instruction(&mut self, location: SourceLocation) -> Result<Token, ParseError> {
        let start = self.pos;

        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }

        let text = &self.source[start..self.pos];

        // Check if it's a label (followed by :)
        if self.peek() == Some(':') {
            self.advance();
            return Ok(Token {
                kind: TokenKind::Label,
                text: self.source[start..self.pos].to_string(),
                location,
            });
        }

        // Check if it's an instruction
        if self.is_instruction(text) {
            // For instructions, read modifiers and operands
            let instr_end = self.pos;
            self.skip_whitespace();

            // Read the rest of the line for operands
            let operand_start = self.pos;
            while let Some(c) = self.peek() {
                if c == '\n' || c == ';' || c == '{' || c == '}' {
                    break;
                }
                self.advance();
            }

            let full_text = if operand_start < self.pos {
                format!("{} {}", &self.source[start..instr_end], self.source[operand_start..self.pos].trim())
            } else {
                self.source[start..instr_end].to_string()
            };

            return Ok(Token {
                kind: TokenKind::Instruction,
                text: full_text,
                location,
            });
        }

        Ok(Token {
            kind: TokenKind::Identifier,
            text: text.to_string(),
            location,
        })
    }

    fn is_instruction(&self, text: &str) -> bool {
        matches!(text,
            "ld" | "st" | "mov" | "add" | "sub" | "mul" | "div" | "rem" |
            "mad" | "fma" | "neg" | "abs" | "min" | "max" |
            "and" | "or" | "xor" | "not" | "shl" | "shr" |
            "setp" | "selp" | "cvt" | "cvta" |
            "bra" | "call" | "ret" | "exit" |
            "bar" | "membar" | "atom" | "red" |
            "tex" | "tld4" | "suld" | "sust" |
            "shfl" | "vote" | "match" |
            "mma" | "wmma" | "ldmatrix" |
            "cp" | "prefetch" | "prefetchu"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lex_directives() {
        let ptx = ".version 8.0\n.target sm_70\n.address_size 64";
        let mut lexer = Lexer::new(ptx);

        let tok = lexer.next_token().unwrap();
        assert_eq!(tok.kind, TokenKind::Directive);
        assert!(tok.text.contains(".version"), "Expected .version, got: {}", tok.text);
        assert!(tok.text.contains("8.0"), "Expected 8.0, got: {}", tok.text);

        let tok = lexer.next_token().unwrap();
        assert_eq!(tok.kind, TokenKind::Directive);
        assert!(tok.text.contains(".target"), "Expected .target, got: {}", tok.text);

        let tok = lexer.next_token().unwrap();
        assert_eq!(tok.kind, TokenKind::Directive);
        assert!(tok.text.contains(".address_size"), "Expected .address_size, got: {}", tok.text);
    }

    #[test]
    fn lex_entry_keyword() {
        let ptx = ".entry test_kernel";
        let mut lexer = Lexer::new(ptx);

        let tok = lexer.next_token().unwrap();
        assert_eq!(tok.kind, TokenKind::Entry);
    }

    #[test]
    fn lex_registers() {
        let ptx = "%r0 %rd1 %f2";
        let mut lexer = Lexer::new(ptx);

        let tok = lexer.next_token().unwrap();
        assert_eq!(tok.kind, TokenKind::Identifier);
        assert_eq!(tok.text, "%r0");

        let tok = lexer.next_token().unwrap();
        assert_eq!(tok.kind, TokenKind::Identifier);
        assert_eq!(tok.text, "%rd1");

        let tok = lexer.next_token().unwrap();
        assert_eq!(tok.kind, TokenKind::Identifier);
        assert_eq!(tok.text, "%f2");
    }

    #[test]
    fn lex_instructions() {
        let ptx = "ld.shared.u32 %r0, [%r1]";
        let mut lexer = Lexer::new(ptx);

        let tok = lexer.next_token().unwrap();
        assert_eq!(tok.kind, TokenKind::Instruction);
        assert!(tok.text.starts_with("ld"));
    }

    #[test]
    fn lex_numbers() {
        let ptx = "42 0x1234 3.14";
        let mut lexer = Lexer::new(ptx);

        let tok = lexer.next_token().unwrap();
        assert_eq!(tok.kind, TokenKind::Integer);
        assert_eq!(tok.text, "42");

        let tok = lexer.next_token().unwrap();
        assert_eq!(tok.kind, TokenKind::Integer);
        assert_eq!(tok.text, "0x1234");

        let tok = lexer.next_token().unwrap();
        assert_eq!(tok.kind, TokenKind::Float);
        assert_eq!(tok.text, "3.14");
    }

    #[test]
    fn lex_comments() {
        let ptx = "mov.u32 %r0, 0 // comment\nmov.u32 %r1, 1";
        let mut lexer = Lexer::new(ptx);

        let tok = lexer.next_token().unwrap();
        assert_eq!(tok.kind, TokenKind::Instruction);

        // Comment should be skipped
        let tok = lexer.next_token().unwrap();
        assert_eq!(tok.kind, TokenKind::Instruction);
    }

    #[test]
    fn lex_labels() {
        let ptx = "loop_start:";
        let mut lexer = Lexer::new(ptx);

        let tok = lexer.next_token().unwrap();
        assert_eq!(tok.kind, TokenKind::Label);
        assert!(tok.text.contains("loop_start"));
    }
}
