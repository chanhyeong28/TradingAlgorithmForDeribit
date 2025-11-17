"""
Expiration Selector Module

Automatically selects option expirations based on trading rules:
1. Daily options this week (Mon-Fri)
2. Weekly options this month (every Friday)
3. Monthly options in next 2 months (last Friday)
4. Quarterly options within a year (last Friday of each quarter)
"""

from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import calendar
import logging

logger = logging.getLogger(__name__)


class ExpirationSelector:
    """
    Automatically selects option expirations based on trading rules.
    """
    
    def __init__(self, reference_date: datetime = None):
        """
        Initialize expiration selector
        
        Args:
            reference_date: Reference date (defaults to today)
        """
        self.reference_date = reference_date or datetime.now()
        self.reference_date = self.reference_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    def get_last_friday_of_month(self, year: int, month: int) -> datetime:
        """Get the last Friday of a given month"""
        # Find the last day of the month
        last_day = calendar.monthrange(year, month)[1]
        last_date = datetime(year, month, last_day)
        
        # Go backwards to find the last Friday
        days_back = (last_date.weekday() - calendar.FRIDAY) % 7
        if days_back == 0 and last_date.weekday() != calendar.FRIDAY:
            days_back = 7
        
        last_friday = last_date - timedelta(days=days_back)
        return last_friday.replace(hour=0, minute=0, second=0, microsecond=0)
    
    def get_next_weekday(self, date: datetime, target_weekday: int) -> datetime:
        """
        Get next occurrence of target weekday
        
        Args:
            date: Starting date
            target_weekday: 0=Monday, 1=Tuesday, ..., 6=Sunday
        """
        days_ahead = target_weekday - date.weekday()
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        return date + timedelta(days=days_ahead)
    
    def get_daily_options_this_week(self) -> List[datetime]:
        """
        Get daily options for this week (Monday to Friday).
        If today is a weekend, get next week's daily options.
        Includes today if it's a weekday.
        """
        today = self.reference_date
        today_weekday = today.weekday()
        
        # If today is Saturday or Sunday, start from next Monday
        if today_weekday >= 5:  # Saturday (5) or Sunday (6)
            days_until_monday = 7 - today_weekday
            week_start = today + timedelta(days=days_until_monday)
        else:
            # Today is a weekday, use this week
            week_start = today - timedelta(days=today_weekday)  # Monday of this week
        
        daily_expirations = []
        
        # Monday to Friday of the target week
        for i in range(5):
            exp_date = week_start + timedelta(days=i)
            # Only include future dates (or today)
            if exp_date >= today:
                daily_expirations.append(exp_date)
        
        return sorted(daily_expirations)
    
    def get_weekly_options_this_month(self) -> List[datetime]:
        """
        Get weekly options for this month (every Friday).
        """
        today = self.reference_date
        current_month = today.month
        current_year = today.year
        
        # Find first Friday of current month
        first_day = datetime(current_year, current_month, 1)
        first_friday = self.get_next_weekday(first_day, calendar.FRIDAY)
        
        weekly_expirations = []
        current_date = first_friday
        
        # Collect all Fridays in this month
        while current_date.month == current_month:
            if current_date >= today:  # Only future dates
                weekly_expirations.append(current_date)
            current_date += timedelta(days=7)
        
        return sorted(weekly_expirations)
    
    def get_monthly_options_next_months(self, months_ahead: int = 2) -> List[datetime]:
        """
        Get monthly options for next N months (last Friday of each month).
        
        Args:
            months_ahead: Number of months ahead to include (default: 2)
        """
        today = self.reference_date
        monthly_expirations = []
        
        current_month = today.month
        current_year = today.year
        
        for i in range(months_ahead):
            # Calculate month and year
            month = current_month + i
            year = current_year
            
            # Handle year rollover
            while month > 12:
                month -= 12
                year += 1
            
            last_friday = self.get_last_friday_of_month(year, month)
            
            # Only include future dates
            if last_friday >= today:
                monthly_expirations.append(last_friday)
        
        return sorted(monthly_expirations)
    
    def get_quarterly_options_within_year(self) -> List[datetime]:
        """
        Get quarterly options within a year (last Friday of each quarter).
        Quarters: Dec, Mar, Jun, Sep (last Friday of each quarter-end month)
        """
        today = self.reference_date
        quarterly_expirations = []
        
        current_year = today.year
        current_month = today.month
        
        # Quarters end in: Dec (Q4), Mar (Q1), Jun (Q2), Sep (Q3)
        # Collect quarters from current date to end of next year
        quarters = []
        
        # Start from current year
        start_year = current_year
        
        # If we're past December, start from next year
        if current_month >= 12:
            start_year = current_year + 1
        
        # Collect quarters for this year and next year
        for year in [start_year, start_year + 1]:
            quarters.extend([
                (year, 3),   # March (Q1)
                (year, 6),   # June (Q2)
                (year, 9),   # September (Q3)
                (year, 12),  # December (Q4)
            ])
        
        for year, month in quarters:
            last_friday = self.get_last_friday_of_month(year, month)
            
            # Only include future dates
            if last_friday >= today:
                quarterly_expirations.append(last_friday)
        
        # Remove duplicates and sort, then limit to within a year
        quarterly_expirations = sorted(set(quarterly_expirations))
        
        # Filter to only include expirations within 1 year from today
        one_year_from_today = today + timedelta(days=365)
        quarterly_expirations = [exp for exp in quarterly_expirations if exp <= one_year_from_today]
        
        return quarterly_expirations
    
    def get_all_expirations(self) -> Dict[str, List[datetime]]:
        """
        Get all expirations grouped by category.
        
        Returns:
            Dictionary with keys: 'daily', 'weekly', 'monthly', 'quarterly'
        """
        daily = self.get_daily_options_this_week()
        weekly = self.get_weekly_options_this_month()
        monthly = self.get_monthly_options_next_months(months_ahead=2)
        quarterly = self.get_quarterly_options_within_year()
        
        return {
            'daily': daily,
            'weekly': weekly,
            'monthly': monthly,
            'quarterly': quarterly
        }
    
    def get_all_expirations_flat(self) -> List[datetime]:
        """Get all expirations as a flat sorted list (no duplicates)"""
        all_exp = self.get_all_expirations()
        # Combine all and remove duplicates
        flat_list = []
        seen = set()
        
        for category_expirations in all_exp.values():
            for exp in category_expirations:
                exp_key = exp.date()
                if exp_key not in seen:
                    seen.add(exp_key)
                    flat_list.append(exp)
        
        return sorted(flat_list)
    
    def _format_expiration_date(self, date: datetime) -> str:
        """
        Format expiration date in Deribit format: "3NOV25" or "26DEC25"
        Single-digit days have no leading zero.
        
        Args:
            date: Datetime object
            
        Returns:
            Formatted string like "3NOV25" or "26DEC25"
        """
        day = int(date.strftime('%d'))  # Remove leading zero
        month_year = date.strftime('%b%y').upper()
        return f"{day}{month_year}"
    
    def get_expirations_english_format(self) -> List[str]:
        """
        Get all expirations in Deribit format: "26DEC25" or "3NOV25"
        Single-digit days have no leading zero.
        
        Returns:
            List of expiration strings sorted by date
        """
        expirations = self.get_all_expirations_flat()
        return [self._format_expiration_date(exp) for exp in expirations]
    
    def get_futures_map(self) -> Dict[str, str]:
        """
        Generate futures map for expirations.
        Maps expiration to future name format.
        
        Note: Daily options don't have futures, so they are excluded.
        They should use BTC-PERPETUAL as underlying.
        
        Returns:
            Dictionary mapping expiration string to future format
        """
        all_exp = self.get_all_expirations()
        futures_map = {}
        
        # Only create futures maps for non-daily expirations
        # Daily options should use perpetual
        non_daily_expirations = []
        non_daily_expirations.extend(all_exp['weekly'])
        non_daily_expirations.extend(all_exp['monthly'])
        non_daily_expirations.extend(all_exp['quarterly'])
        
        for exp in non_daily_expirations:
            exp_str = self._format_expiration_date(exp)
            # Format: "26DEC25" -> "26DEC25(futures)"
            futures_map[exp_str] = f"{exp_str}(futures)"
        
        return futures_map
    
    def get_daily_expirations_english(self) -> List[str]:
        """
        Get daily expirations in English format.
        These don't have futures and should use perpetual.
        
        Returns:
            List of daily expiration strings
        """
        daily_exp = self.get_daily_options_this_week()
        return [self._format_expiration_date(exp) for exp in daily_exp]
    
    def get_summary(self) -> Dict:
        """Get summary of selected expirations"""
        all_exp = self.get_all_expirations()
        flat_list = self.get_all_expirations_flat()
        
        return {
            'reference_date': self.reference_date.strftime('%Y-%m-%d'),
            'daily_count': len(all_exp['daily']),
            'weekly_count': len(all_exp['weekly']),
            'monthly_count': len(all_exp['monthly']),
            'quarterly_count': len(all_exp['quarterly']),
            'total_count': len(flat_list),
            'expirations': {
                'daily': [self._format_expiration_date(e) for e in all_exp['daily']],
                'weekly': [self._format_expiration_date(e) for e in all_exp['weekly']],
                'monthly': [self._format_expiration_date(e) for e in all_exp['monthly']],
                'quarterly': [self._format_expiration_date(e) for e in all_exp['quarterly']],
            },
            'all_expirations': [self._format_expiration_date(e) for e in flat_list]
        }

