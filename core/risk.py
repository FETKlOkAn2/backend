import time
import logging
from datetime import datetime

class Risk_Handler:
    """
    Enhanced Risk Handler for cryptocurrency trading bot
    Handles position sizing, risk management, and drawdown protection
    """
    def __init__(self, client=None, risk_percent=0.02, max_drawdown=0.15, max_open_trades=3):
        """
        Initialize the Risk Handler
        
        Args:
            client: Trading client API connection
            risk_percent: Percentage of account to risk per trade (default: 2%)
            max_drawdown: Maximum allowed drawdown before reducing position sizes (default: 15%)
            max_open_trades: Maximum number of concurrent open trades allowed (default: 3)
        """
        # Client connection
        self.client = client
        
        # Risk parameters
        self.risk_percent = risk_percent
        self.max_drawdown = max_drawdown
        self.max_open_trades = max_open_trades
        self.initial_balance = 1000
        self.total_balance = self.initial_balance
        self.balance_high_watermark = self.initial_balance
        self.current_drawdown = 0
        self.active_trades = {}
        
        # Symbol-specific parameters
        self.symbol_params = {}
        
        # Get actual balance if client provided
        if client is not None:
            try:
                self.total_balance = self.client.get_account_balance()
                self.initial_balance = self.total_balance
                self.balance_high_watermark = self.total_balance
                logging.info(f"Initial balance: {self.total_balance}")
            except Exception as e:
                logging.error(f"Error fetching balance: {str(e)}")
        
        logging.info(f"Risk Handler Initialized - Risk per trade: {self.risk_percent*100}%, Max drawdown: {self.max_drawdown*100}%")
        if client is not None:
            logging.info(f"Client: {self.client.__class__.__name__}")
    
    def calculate_position_size(self, symbol, entry_price, stop_loss_price):
        """
        Calculate optimal position size based on risk parameters
        
        Args:
            symbol: Trading pair symbol
            entry_price: Planned entry price
            stop_loss_price: Planned stop loss price
            
        Returns:
            float: Position size in base currency
        """
        if entry_price <= 0 or stop_loss_price <= 0:
            logging.warning("Invalid prices provided for position sizing")
            return 0
            
        # Calculate risk amount based on current balance
        risk_amount = self.total_balance * self.risk_percent
        
        # Apply drawdown protection
        if self.current_drawdown > self.max_drawdown / 2:
            reduction_factor = 1 - (self.current_drawdown / self.max_drawdown)
            risk_amount *= max(0.5, reduction_factor)
            logging.info(f"Reducing position size due to drawdown of {self.current_drawdown*100:.2f}%")
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss_price) / entry_price
        
        # Prevent division by zero
        if risk_per_unit == 0:
            logging.warning("Stop loss too close to entry price")
            return 0
            
        # Calculate position size
        position_size = risk_amount / risk_per_unit
        
        # Apply symbol-specific limits if available
        if symbol in self.symbol_params:
            min_size = self.symbol_params[symbol].get('min_size', 0)
            max_size = self.symbol_params[symbol].get('max_size', float('inf'))
            step_size = self.symbol_params[symbol].get('step_size', 0.00001)
            
            # Round to nearest step size
            position_size = round(position_size / step_size) * step_size
            
            # Ensure within limits
            position_size = max(min_size, min(position_size, max_size))
        
        # Apply maximum open trades limit
        if len(self.active_trades) >= self.max_open_trades:
            logging.warning(f"Maximum open trades limit reached ({self.max_open_trades})")
            return 0
            
        return position_size
    
    def update_balance(self, new_balance=None):
        """
        Update account balance and recalculate drawdown
        
        Args:
            new_balance: New balance value, or None to fetch from client
        """
        previous_balance = self.total_balance
        
        if new_balance is not None:
            self.total_balance = new_balance
        elif self.client is not None:
            try:
                self.total_balance = self.client.get_account_balance()
            except Exception as e:
                logging.error(f"Error updating balance: {str(e)}")
                return
        
        # Update high watermark
        if self.total_balance > self.balance_high_watermark:
            self.balance_high_watermark = self.total_balance
            
        # Calculate current drawdown
        if self.balance_high_watermark > 0:
            self.current_drawdown = 1 - (self.total_balance / self.balance_high_watermark)
            
        logging.info(f"Balance updated: {previous_balance} → {self.total_balance}")
        logging.info(f"Current drawdown: {self.current_drawdown*100:.2f}%")
        
        # Check if max drawdown exceeded
        if self.current_drawdown >= self.max_drawdown:
            logging.warning(f"Maximum drawdown of {self.max_drawdown*100}% exceeded! Current: {self.current_drawdown*100:.2f}%")
            return True  # Signal that max drawdown has been exceeded
        return False
    
    def register_trade(self, trade_id, symbol, entry_price, position_size, stop_loss_price, take_profit_price=None):
        """
        Register a new active trade
        
        Args:
            trade_id: Unique trade identifier
            symbol: Trading pair symbol
            entry_price: Entry price
            position_size: Position size
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price (optional)
        """
        self.active_trades[trade_id] = {
            'symbol': symbol,
            'entry_time': datetime.now(),
            'entry_price': entry_price,
            'position_size': position_size,
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'risk_amount': self.total_balance * self.risk_percent
        }
        logging.info(f"Trade registered: {trade_id} on {symbol} with {position_size} units")
    
    def close_trade(self, trade_id, exit_price):
        """
        Close an active trade and calculate profit/loss
        
        Args:
            trade_id: Unique trade identifier
            exit_price: Exit price
            
        Returns:
            float: Profit/loss amount
        """
        if trade_id not in self.active_trades:
            logging.warning(f"Trade {trade_id} not found in active trades")
            return 0
            
        trade = self.active_trades[trade_id]
        profit_loss = (exit_price - trade['entry_price']) * trade['position_size']
        
        logging.info(f"Trade closed: {trade_id} with P/L: {profit_loss}")
        del self.active_trades[trade_id]
        
        # Update balance
        self.update_balance(self.total_balance + profit_loss)
        
        return profit_loss
    
    def add_symbol_params(self, symbol, min_size, max_size, step_size):
        """
        Add or update symbol-specific trading parameters
        
        Args:
            symbol: Trading pair symbol
            min_size: Minimum order size
            max_size: Maximum order size
            step_size: Size increment
        """
        self.symbol_params[symbol] = {
            'min_size': min_size,
            'max_size': max_size,
            'step_size': step_size
        }
        logging.info(f"Added parameters for {symbol}: min={min_size}, max={max_size}, step={step_size}")
    
    def get_total_exposure(self):
        """
        Calculate total exposure across all active trades
        
        Returns:
            float: Total exposure as percentage of account
        """
        if not self.active_trades:
            return 0
            
        total_risk = sum(trade['risk_amount'] for trade in self.active_trades.values())
        return total_risk / self.total_balance