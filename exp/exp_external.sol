pragma solidity ^0.5.0;

// 合约A
contract ContractA {
    uint256 public publicVariable;

    constructor(uint256 _initialValue) public{
        publicVariable = _initialValue;
    }
}

// 外部合约B
contract ContractB {
    // 合约A的实例
    ContractA public contractAInstance;

    constructor(address _contractAAddress) public{
        // 在构造函数中传入合约A的地址，创建合约A的实例
        contractAInstance = ContractA(_contractAAddress);
    }

    // 外部合约B调用合约A的公共变量
    function getContractAPublicVariable() public view returns (uint256) {
        // 调用合约A的公共变量的自动生成的getter函数
        return contractAInstance.publicVariable();
    }
}