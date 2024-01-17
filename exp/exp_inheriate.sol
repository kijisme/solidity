pragma solidity ^0.5.0;

// 父合约
contract ParentContract {
    uint256 public parentVariable;

    constructor(uint256 _value) public{
        parentVariable = _value;
    }

    function getParentFunction() public view returns (string memory) {
        return "This is a function from ParentContract";
    }
}

// 子合约继承父合约
contract ChildContract is ParentContract {
    uint256 public childVariable;

    constructor(uint256 _parentValue, uint256 _childValue) public ParentContract(_parentValue) {
        childVariable = _childValue;
    }

    function getChildFunction() public view returns (string memory) {
        return "This is a function from ChildContract";
    }
}
