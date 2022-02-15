# singleton opcode classes
_OPCODE_PLACEHOLDER   = {'placeholder'}
_OPCODE_OUTPUT        = {'output'}
_OPCODE_CALL_MODULE   = {'call_module'}
_OPCODE_CALL_FUNCTION = {'call_function'}
_OPCODE_CALL_METHOD   = {'call_method'}

# higher-level opcode classes
_OPCODES_IO         = _OPCODE_PLACEHOLDER   | _OPCODE_OUTPUT
_OPCODES_NONMODULAR = _OPCODE_CALL_FUNCTION | _OPCODE_CALL_METHOD
