{"EIP20.sol":{"content":"/*\nImplements EIP20 token standard: https://github.com/ethereum/EIPs/blob/master/EIPS/eip-20.md\n.*/\n\npragma solidity ^0.4.21;\n\nimport \"./EIP20Interface.sol\";\n\ncontract EIP20 is EIP20Interface {\n\n    uint256 constant private MAX_UINT256 = 2**256 - 1;\n    mapping (address =\u003e uint256) public balances;\n    mapping (address =\u003e mapping (address =\u003e uint256)) public allowed;\n\n    /*\n    NOTE:\n    The following variables are OPTIONAL vanities. One does not have to include them.\n    They allow one to customise the token contract \u0026 in no way influences the core functionality.\n    Some wallets/interfaces might not even bother to look at this information.\n    */\n    string public name;                   //fancy name: eg Simon Bucks\n    uint8 public decimals;                //How many decimals to show.\n    string public symbol;                 //An identifier: eg SBX\n\n    function EIP20(uint256 _initialAmount, string _tokenName, uint8 _decimalUnits, string _tokenSymbol) public {\n\n        balances[msg.sender] = _initialAmount;               // Give the creator all initial tokens\n        totalSupply = _initialAmount;                        // Update total supply\n        name = _tokenName;                                   // Set the name for display purposes\n        decimals = _decimalUnits;                            // Amount of decimals for display purposes\n        symbol = _tokenSymbol;                               // Set the symbol for display purposes\n    }\n\n    function transfer(address _to, uint256 _value) public returns (bool success) {\n        require(balances[msg.sender] \u003e= _value);\n        balances[msg.sender] -= _value;\n        balances[_to] += _value;\n        emit Transfer(msg.sender, _to, _value); //solhint-disable-line indent, no-unused-vars\n        return true;\n    }\n\n    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {\n        uint256 allowance = allowed[_from][msg.sender];\n        require(balances[_from] \u003e= _value \u0026\u0026 allowance \u003e= _value);\n        balances[_to] += _value;\n        balances[_from] -= _value;\n        if (allowance \u003c MAX_UINT256) {\n            allowed[_from][msg.sender] -= _value;\n        }\n        emit Transfer(_from, _to, _value); //solhint-disable-line indent, no-unused-vars\n        return true;\n    }\n\n    function balanceOf(address _owner) public view returns (uint256 balance) {\n        return balances[_owner];\n    }\n\n    function approve(address _spender, uint256 _value) public returns (bool success) {\n        allowed[msg.sender][_spender] = _value;\n        emit Approval(msg.sender, _spender, _value); //solhint-disable-line indent, no-unused-vars\n        return true;\n    }\n\n    function allowance(address _owner, address _spender) public view returns (uint256 remaining) {\n        return allowed[_owner][_spender];\n    }\n}\n"},"EIP20Factory.sol":{"content":"import \"./EIP20.sol\";\n\npragma solidity ^0.4.21;\n\n\ncontract EIP20Factory {\n\n    mapping(address =\u003e address[]) public created;\n    mapping(address =\u003e bool) public isEIP20; //verify without having to do a bytecode check.\n    bytes public EIP20ByteCode; // solhint-disable-line var-name-mixedcase\n\n    function EIP20Factory() public {\n        //upon creation of the factory, deploy a EIP20 (parameters are meaningless) and store the bytecode provably.\n        address verifiedToken = createEIP20(10000, \"Verify Token\", 3, \"VTX\");\n        EIP20ByteCode = codeAt(verifiedToken);\n    }\n\n    //verifies if a contract that has been deployed is a Human Standard Token.\n    //NOTE: This is a very expensive function, and should only be used in an eth_call. ~800k gas\n    function verifyEIP20(address _tokenContract) public view returns (bool) {\n        bytes memory fetchedTokenByteCode = codeAt(_tokenContract);\n\n        if (fetchedTokenByteCode.length != EIP20ByteCode.length) {\n            return false; //clear mismatch\n        }\n\n      //starting iterating through it if lengths match\n        for (uint i = 0; i \u003c fetchedTokenByteCode.length; i++) {\n            if (fetchedTokenByteCode[i] != EIP20ByteCode[i]) {\n                return false;\n            }\n        }\n        return true;\n    }\n\n    function createEIP20(uint256 _initialAmount, string _name, uint8 _decimals, string _symbol)\n        public\n    returns (address) {\n\n        EIP20 newToken = (new EIP20(_initialAmount, _name, _decimals, _symbol));\n        created[msg.sender].push(address(newToken));\n        isEIP20[address(newToken)] = true;\n        //the factory will own the created tokens. You must transfer them.\n        newToken.transfer(msg.sender, _initialAmount);\n        return address(newToken);\n    }\n\n    //for now, keeping this internal. Ideally there should also be a live version of this that\n    // any contract can use, lib-style.\n    //retrieves the bytecode at a specific address.\n    function codeAt(address _addr) internal view returns (bytes outputCode) {\n        assembly { // solhint-disable-line no-inline-assembly\n            // retrieve the size of the code, this needs assembly\n            let size := extcodesize(_addr)\n            // allocate output byte array - this could also be done without assembly\n            // by using outputCode = new bytes(size)\n            outputCode := mload(0x40)\n            // new \"memory end\" including padding\n            mstore(0x40, add(outputCode, and(add(add(size, 0x20), 0x1f), not(0x1f))))\n            // store length in memory\n            mstore(outputCode, size)\n            // actually retrieve the code, this needs assembly\n            extcodecopy(_addr, add(outputCode, 0x20), 0, size)\n        }\n    }\n}\n"},"EIP20Interface.sol":{"content":"// Abstract contract for the full ERC 20 Token standard\n// https://github.com/ethereum/EIPs/blob/master/EIPS/eip-20.md\npragma solidity ^0.4.21;\n\n\ncontract EIP20Interface {\n    /* This is a slight change to the ERC20 base standard.\n    function totalSupply() constant returns (uint256 supply);\n    is replaced with:\n    uint256 public totalSupply;\n    This automatically creates a getter function for the totalSupply.\n    This is moved to the base contract since public getter functions are not\n    currently recognised as an implementation of the matching abstract\n    function by the compiler.\n    */\n    /// total amount of tokens\n    uint256 public totalSupply;\n\n    /// @param _owner The address from which the balance will be retrieved\n    /// @return The balance\n    function balanceOf(address _owner) public view returns (uint256 balance);\n\n    /// @notice send `_value` token to `_to` from `msg.sender`\n    /// @param _to The address of the recipient\n    /// @param _value The amount of token to be transferred\n    /// @return Whether the transfer was successful or not\n    function transfer(address _to, uint256 _value) public returns (bool success);\n\n    /// @notice send `_value` token to `_to` from `_from` on the condition it is approved by `_from`\n    /// @param _from The address of the sender\n    /// @param _to The address of the recipient\n    /// @param _value The amount of token to be transferred\n    /// @return Whether the transfer was successful or not\n    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success);\n\n    /// @notice `msg.sender` approves `_spender` to spend `_value` tokens\n    /// @param _spender The address of the account able to transfer the tokens\n    /// @param _value The amount of tokens to be approved for transfer\n    /// @return Whether the approval was successful or not\n    function approve(address _spender, uint256 _value) public returns (bool success);\n\n    /// @param _owner The address of the account owning tokens\n    /// @param _spender The address of the account able to transfer the tokens\n    /// @return Amount of remaining tokens allowed to spent\n    function allowance(address _owner, address _spender) public view returns (uint256 remaining);\n\n    // solhint-disable-next-line no-simple-event-func-name\n    event Transfer(address indexed _from, address indexed _to, uint256 _value);\n    event Approval(address indexed _owner, address indexed _spender, uint256 _value);\n}\n"}}