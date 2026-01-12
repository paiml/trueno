//! PTX Parser Module
//!
//! Provides lexing and parsing of PTX source code into a typed AST.

mod lexer;
mod ast;
pub mod types;
mod error;

pub use lexer::{Lexer, Token, TokenKind};
pub use ast::{
    PtxModule, KernelDef, FunctionDef, GlobalDecl, Param, RegisterDecl, SharedMemDecl,
    Statement, Instruction, Directive, Operand, Predicate, SourceLocation,
};
pub use types::{PtxType, AddressSpace, SmTarget, Opcode, Modifier};
pub use error::ParseError;

/// PTX Parser - constructs AST from token stream
pub struct Parser<'a> {
    lexer: Lexer<'a>,
    current: Token,
    peek: Token,
}

impl<'a> Parser<'a> {
    /// Create a new parser from PTX source
    pub fn new(source: &'a str) -> Result<Self, ParseError> {
        let mut lexer = Lexer::new(source);
        let current = lexer.next_token()?;
        let peek = lexer.next_token()?;
        Ok(Self { lexer, current, peek })
    }

    /// Parse the PTX source into a module
    pub fn parse(&mut self) -> Result<PtxModule, ParseError> {
        let mut module = PtxModule::default();

        while self.current.kind != TokenKind::Eof {
            match self.current.kind {
                TokenKind::Directive => {
                    self.parse_directive(&mut module)?;
                }
                TokenKind::Entry | TokenKind::Func => {
                    let kernel = self.parse_kernel()?;
                    module.kernels.push(kernel);
                }
                _ => {
                    self.advance()?;
                }
            }
        }

        Ok(module)
    }

    fn advance(&mut self) -> Result<(), ParseError> {
        self.current = std::mem::replace(&mut self.peek, self.lexer.next_token()?);
        Ok(())
    }

    fn parse_directive(&mut self, module: &mut PtxModule) -> Result<(), ParseError> {
        let directive_text = self.current.text.clone();

        if directive_text.starts_with(".version") {
            module.version = self.parse_version_directive()?;
        } else if directive_text.starts_with(".target") {
            module.target = self.parse_target_directive()?;
        } else if directive_text.starts_with(".address_size") {
            module.address_size = self.parse_address_size_directive()?;
        }

        self.advance()?;
        Ok(())
    }

    fn parse_version_directive(&self) -> Result<(u8, u8), ParseError> {
        // Parse ".version X.Y"
        let text = &self.current.text;
        if let Some(rest) = text.strip_prefix(".version") {
            let version_str = rest.trim();
            let parts: Vec<&str> = version_str.split('.').collect();
            if parts.len() >= 2 {
                let major = parts[0].parse().unwrap_or(0);
                let minor = parts[1].parse().unwrap_or(0);
                return Ok((major, minor));
            }
        }
        Ok((0, 0))
    }

    fn parse_target_directive(&self) -> Result<SmTarget, ParseError> {
        let text = &self.current.text;
        if text.contains("sm_70") {
            Ok(SmTarget::Sm70)
        } else if text.contains("sm_75") {
            Ok(SmTarget::Sm75)
        } else if text.contains("sm_80") {
            Ok(SmTarget::Sm80)
        } else if text.contains("sm_86") {
            Ok(SmTarget::Sm86)
        } else if text.contains("sm_89") {
            Ok(SmTarget::Sm89)
        } else if text.contains("sm_90") {
            Ok(SmTarget::Sm90)
        } else {
            Ok(SmTarget::Unknown)
        }
    }

    fn parse_address_size_directive(&self) -> Result<u8, ParseError> {
        let text = &self.current.text;
        if text.contains("64") {
            Ok(64)
        } else if text.contains("32") {
            Ok(32)
        } else {
            Ok(0)
        }
    }

    fn parse_kernel(&mut self) -> Result<KernelDef, ParseError> {
        let is_entry = self.current.kind == TokenKind::Entry;
        self.advance()?;

        // Parse kernel name
        let name = if self.current.kind == TokenKind::Identifier {
            let n = self.current.text.clone();
            self.advance()?;
            n
        } else {
            return Err(ParseError::UnexpectedToken {
                expected: "kernel name".into(),
                found: format!("{:?}", self.current.kind),
                location: self.current.location.clone(),
            });
        };

        let mut kernel = KernelDef {
            name,
            is_entry,
            params: Vec::new(),
            registers: Vec::new(),
            shared_mem: Vec::new(),
            body: Vec::new(),
        };

        // Parse parameter list and body (simplified)
        while self.current.kind != TokenKind::Eof {
            if self.current.kind == TokenKind::Entry || self.current.kind == TokenKind::Func {
                break;
            }

            match self.current.kind {
                TokenKind::Reg => {
                    let reg = self.parse_register_decl()?;
                    kernel.registers.push(reg);
                }
                TokenKind::Shared => {
                    let shared = self.parse_shared_mem_decl()?;
                    kernel.shared_mem.push(shared);
                }
                TokenKind::Instruction => {
                    let instr = self.parse_instruction()?;
                    kernel.body.push(Statement::Instruction(instr));
                }
                TokenKind::Label => {
                    let label = self.current.text.trim_end_matches(':').to_string();
                    kernel.body.push(Statement::Label(label));
                    self.advance()?;
                }
                _ => {
                    self.advance()?;
                }
            }
        }

        Ok(kernel)
    }

    fn parse_register_decl(&mut self) -> Result<RegisterDecl, ParseError> {
        // Parse ".reg .TYPE %name"
        let text = self.current.text.clone();
        self.advance()?;

        // Extract type and name from the declaration
        let (ty, name) = self.extract_reg_type_and_name(&text);

        Ok(RegisterDecl { name, ty })
    }

    fn extract_reg_type_and_name(&self, text: &str) -> (PtxType, String) {
        let ty = if text.contains(".b64") || text.contains(".u64") {
            PtxType::B64
        } else if text.contains(".b32") || text.contains(".u32") {
            PtxType::U32
        } else if text.contains(".f32") {
            PtxType::F32
        } else if text.contains(".f64") {
            PtxType::F64
        } else if text.contains(".pred") {
            PtxType::Pred
        } else {
            PtxType::B32
        };

        // Extract register name (starts with %)
        let name = text.split_whitespace()
            .find(|s| s.starts_with('%'))
            .map(|s| s.trim_end_matches(|c: char| !c.is_alphanumeric() && c != '%' && c != '_'))
            .unwrap_or("%unknown")
            .to_string();

        (ty, name)
    }

    fn parse_shared_mem_decl(&mut self) -> Result<SharedMemDecl, ParseError> {
        let text = self.current.text.clone();
        self.advance()?;

        // Parse shared memory declaration
        let name = text.split_whitespace()
            .find(|s| !s.starts_with('.'))
            .unwrap_or("unknown")
            .to_string();

        let size = text.split('[')
            .nth(1)
            .and_then(|s| s.split(']').next())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        Ok(SharedMemDecl { name, size, ty: PtxType::B8 })
    }

    fn parse_instruction(&mut self) -> Result<Instruction, ParseError> {
        let text = self.current.text.clone();
        let location = self.current.location.clone();
        self.advance()?;

        let (opcode, modifiers) = self.parse_opcode(&text);
        let operands = self.parse_operands(&text);

        Ok(Instruction {
            opcode,
            modifiers,
            operands,
            predicate: None,
            location,
        })
    }

    fn parse_opcode(&self, text: &str) -> (Opcode, Vec<Modifier>) {
        let parts: Vec<&str> = text.split_whitespace().next()
            .unwrap_or("")
            .split('.')
            .collect();

        let opcode = match parts.first().copied() {
            Some("ld") => Opcode::Ld,
            Some("st") => Opcode::St,
            Some("mov") => Opcode::Mov,
            Some("add") => Opcode::Add,
            Some("sub") => Opcode::Sub,
            Some("mul") => Opcode::Mul,
            Some("mad") => Opcode::Mad,
            Some("fma") => Opcode::Fma,
            Some("cvta") => Opcode::Cvta,
            Some("cvt") => Opcode::Cvt,
            Some("setp") => Opcode::Setp,
            Some("bra") => Opcode::Bra,
            Some("bar") => Opcode::Bar,
            Some("atom") => Opcode::Atom,
            Some("ret") => Opcode::Ret,
            Some("exit") => Opcode::Exit,
            Some("and") => Opcode::And,
            Some("or") => Opcode::Or,
            Some("xor") => Opcode::Xor,
            Some("shl") => Opcode::Shl,
            Some("shr") => Opcode::Shr,
            Some("membar") => Opcode::MemBar,
            _ => Opcode::Unknown,
        };

        let modifiers = parts.iter().skip(1)
            .map(|&m| self.parse_modifier(m))
            .collect();

        (opcode, modifiers)
    }

    fn parse_modifier(&self, s: &str) -> Modifier {
        match s {
            "shared" => Modifier::Shared,
            "global" => Modifier::Global,
            "local" => Modifier::Local,
            "const" => Modifier::Const,
            "param" => Modifier::Param,
            "u32" => Modifier::U32,
            "u64" => Modifier::U64,
            "s32" => Modifier::S32,
            "s64" => Modifier::S64,
            "f32" => Modifier::F32,
            "f64" => Modifier::F64,
            "b32" => Modifier::B32,
            "b64" => Modifier::B64,
            "sync" => Modifier::Sync,
            "cta" => Modifier::Cta,
            "gl" => Modifier::Gl,
            "add" => Modifier::AtomicAdd,
            "cas" => Modifier::AtomicCas,
            _ => Modifier::Other(s.to_string()),
        }
    }

    fn parse_operands(&self, text: &str) -> Vec<Operand> {
        // Simple operand parsing - extract tokens after the opcode
        let mut operands = Vec::new();

        // Skip the opcode part
        let operand_part = text.split_whitespace()
            .skip(1)
            .collect::<Vec<_>>()
            .join(" ");

        for part in operand_part.split(',') {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                continue;
            }

            let operand = if trimmed.starts_with('%') {
                Operand::Register(trimmed.to_string())
            } else if trimmed.starts_with('[') {
                Operand::Memory(trimmed.to_string())
            } else if trimmed.parse::<i64>().is_ok() {
                Operand::Immediate(trimmed.parse().unwrap_or(0))
            } else {
                Operand::Label(trimmed.to_string())
            };
            operands.push(operand);
        }

        operands
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // F001: PTX contains .version directive
    #[test]
    fn f001_version_directive_present() {
        let ptx = r#"
            .version 8.0
            .target sm_70
            .address_size 64
        "#;
        let mut parser = Parser::new(ptx).unwrap();
        let module = parser.parse().unwrap();
        assert!(module.version.0 > 0, "F001: Missing .version directive");
    }

    // F001: Negative test
    #[test]
    fn f001_version_directive_missing() {
        let ptx = r#"
            .target sm_70
            .address_size 64
        "#;
        let mut parser = Parser::new(ptx).unwrap();
        let module = parser.parse().unwrap();
        assert_eq!(module.version, (0, 0), "Should detect missing version");
    }

    // F002: PTX contains .target directive
    #[test]
    fn f002_target_directive_present() {
        let ptx = r#"
            .version 8.0
            .target sm_70
            .address_size 64
        "#;
        let mut parser = Parser::new(ptx).unwrap();
        let module = parser.parse().unwrap();
        assert_ne!(module.target, SmTarget::Unknown, "F002: Missing .target directive");
    }

    // F003: address_size is 32 or 64
    #[test]
    fn f003_address_size_valid() {
        let ptx = r#"
            .version 8.0
            .target sm_70
            .address_size 64
        "#;
        let mut parser = Parser::new(ptx).unwrap();
        let module = parser.parse().unwrap();
        assert!(module.address_size == 32 || module.address_size == 64,
            "F003: address_size must be 32 or 64");
    }

    // Test parsing a simple kernel
    #[test]
    fn parse_simple_kernel() {
        let ptx = r#"
            .version 8.0
            .target sm_70
            .address_size 64

            .entry test_kernel(
                .param .u64 param0
            )
            {
                .reg .u32 %r<10>;
                .reg .u64 %rd<10>;

                mov.u32 %r0, 0;
                ret;
            }
        "#;
        let mut parser = Parser::new(ptx).unwrap();
        let module = parser.parse().unwrap();

        assert_eq!(module.version, (8, 0));
        assert_eq!(module.target, SmTarget::Sm70);
        assert_eq!(module.address_size, 64);
        assert!(!module.kernels.is_empty(), "Should have at least one kernel");
    }

    // Test parsing instructions
    #[test]
    fn parse_ld_st_instructions() {
        let ptx = r#"
            .version 8.0
            .target sm_70
            .address_size 64

            .entry test(
            )
            {
                .reg .u32 %r<10>;

                ld.shared.u32 %r0, [%r1];
                st.global.u32 [%r2], %r0;
                ret;
            }
        "#;
        let mut parser = Parser::new(ptx).unwrap();
        let module = parser.parse().unwrap();

        let kernel = &module.kernels[0];
        let instructions: Vec<_> = kernel.body.iter()
            .filter_map(|s| match s {
                Statement::Instruction(i) => Some(i),
                _ => None,
            })
            .collect();

        assert!(instructions.iter().any(|i| matches!(i.opcode, Opcode::Ld)));
        assert!(instructions.iter().any(|i| matches!(i.opcode, Opcode::St)));
    }
}
