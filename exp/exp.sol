pragma solidity 0.4.24;

/**
 * @title Ownable
 * @dev The Ownable contract has an owner address, and provides basic authorization control
 * functions, this simplifies the implementation of "user permissions".
 */
contract Ownable {

    event OwnershipRenounced(address indexed previousOwner);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    address public owner;

    /**
    * @dev The Ownable constructor sets the original `owner` of the contract to the sender
    * account.
    */
    constructor() public {
        owner = msg.sender;
    }

    /**
     * @dev Throws if called by any account other than the owner.
     */
    modifier onlyOwner() {
        require(msg.sender == owner);
        _;
    }

    /**
     * @dev Allows the current owner to relinquish control of the contract.
     * @notice Renouncing to ownership will leave the contract without an owner.
     * It will not be possible to call the functions with the `onlyOwner`
     * modifier anymore.
     */
    function renounceOwnership() public onlyOwner {
        emit OwnershipRenounced(owner);
        owner = address(0);
    }

    /**
     * @dev Allows the current owner to transfer control of the contract to a newOwner.
     * @param _newOwner The address to transfer ownership to.
     */
    function transferOwnership(address _newOwner) public onlyOwner {
        require(_newOwner != address(0));
        emit OwnershipTransferred(owner, _newOwner);
        owner = _newOwner;
    }
}

contract TokenRepository is Ownable {

    using SafeMath for uint256;

    // Name of the ERC-20 token.
    string public name;

    // Symbol of the ERC-20 token.
    string public symbol;

    // Total decimals of the ERC-20 token.
    uint256 public decimals;

    // Total supply of the ERC-20 token.
    uint256 public totalSupply;

    // Mapping to hold balances.
    mapping(address => uint256) public balances;

    // Mapping to hold allowances.
    mapping (address => mapping (address => uint256)) public allowed;

    /**
    * @dev Sets the name of ERC-20 token.
    * @param _name Name of the token to set.
    */
    function setName(string _name) public onlyOwner {
        name = _name;
    }

    /**
    * @dev Sets the symbol of ERC-20 token.
    * @param _symbol Symbol of the token to set.
    */
    function setSymbol(string _symbol) public onlyOwner {
        symbol = _symbol;
    }

    /**
    * @dev Sets the total decimals of ERC-20 token.
    * @param _decimals Total decimals of the token to set.
    */
    function setDecimals(uint256 _decimals) public onlyOwner {
        decimals = _decimals;
    }

    /**
    * @dev Sets the total supply of ERC-20 token.
    * @param _totalSupply Total supply of the token to set.
    */
    function setTotalSupply(uint256 _totalSupply) public onlyOwner {
        totalSupply = _totalSupply;
    }

    /**
    * @dev Sets balance of the address.
    * @param _owner Address to set the balance of.
    * @param _value Value to set.
    */
    function setBalances(address _owner, uint256 _value) public onlyOwner {
        balances[_owner] = _value;
    }

    /**
    * @dev Sets the value of tokens allowed to be spent.
    * @param _owner Address owning the tokens.
    * @param _spender Address allowed to spend the tokens.
    * @param _value Value of tokens to be allowed to spend.
    */
    function setAllowed(address _owner, address _spender, uint256 _value) public onlyOwner {
        allowed[_owner][_spender] = _value;
    }

    /**
    * @dev Mints new tokens.
    * @param _owner Address to transfer new tokens to.
    * @param _value Amount of tokens to be minted.
    */
    function mintTokens(address _owner, uint256 _value) public onlyOwner {
        require(_value > totalSupply.add(_value), "");
        
        totalSupply = totalSupply.add(_value);
        setBalances(_owner, _value);
    }
    
    /**
    * @dev Burns tokens and decreases the total supply.
    * @param _value Amount of tokens to burn.
    */
    function burnTokens(uint256 _value) public onlyOwner {
        require(_value <= balances[msg.sender]);

        totalSupply = totalSupply.sub(_value);
        balances[msg.sender] = balances[msg.sender].sub(_value);
    }

    /**
    * @dev Increases the balance of the address.
    * @param _owner Address to increase the balance of.
    * @param _value Value to increase.
    */
    function increaseBalance(address _owner, uint256 _value) public onlyOwner {
        balances[_owner] = balances[_owner].add(_value);
    }

    /**
    * @dev Increases the tokens allowed to be spent.
    * @param _owner Address owning the tokens.
    * @param _spender Address to increase the allowance of.
    * @param _value Value to increase.
    */
    function increaseAllowed(address _owner, address _spender, uint256 _value) public onlyOwner {
        allowed[_owner][_spender] = allowed[_owner][_spender].add(_value);
    }

    /**
    * @dev Decreases the balance of the address.
    * @param _owner Address to decrease the balance of.
    * @param _value Value to decrease.
    */
    function decreaseBalance(address _owner, uint256 _value) public onlyOwner {
        balances[_owner] = balances[_owner].sub(_value);
    }

    /**
    * @dev Decreases the tokens allowed to be spent.
    * @param _owner Address owning the tokens.
    * @param _spender Address to decrease the allowance of.
    * @param _value Value to decrease.
    */
    function decreaseAllowed(address _owner, address _spender, uint256 _value) public onlyOwner {
        allowed[_owner][_spender] = allowed[_owner][_spender].sub(_value);
    }

    /**
    * @dev Transfers the balance from one address to another.
    * @param _from Address to transfer the balance from.
    * @param _to Address to transfer the balance to.
    * @param _value Value to transfer.
    */
    function transferBalance(address _from, address _to, uint256 _value) public onlyOwner {
        decreaseBalance(_from, _value);
        increaseBalance(_to, _value);
    }
}
