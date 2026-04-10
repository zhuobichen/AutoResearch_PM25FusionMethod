from datetime import datetime, timedelta, timezone
import pytz
import time
import calendar


def timer_decorator(func):
    """
    @description: Timer decorator to measure function execution time
    @param: func: The function to be decorated
    @return: wrapper: Decorator function
    @usage:
    @timer_decorator
    def test():
        print("Hello, world!")
        time.sleep(1)
        print("Goodbye, world!")
    test()
    # Output:
    # Function test execution time: 1.000000 seconds
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} execution time: {end_time - start_time:.6f} seconds")
        return result

    return wrapper


def format_date_to_year_day(date):
    """
    @description: Format date into year and day of the year in YYYYJJJ format, where JJJ is the day of the year (range 001-366).
    For example, 2022001 represents the first day of 2022, and 2022366 represents the last day of 2022.
    @param {str, datetime.datetime, int} date: Date, which can be a datetime.datetime object, or a string in YYYYMMDD or YYYYJJJ format.
    @return {int}: Year and day of the year in YYYYJJJ format
    @example:
    >>> format_date_to_year_day("2022001")
    2022001
    >>> format_date_to_year_day(2022001)
    2022001
    >>> format_date_to_year_day(datetime(2022, 1, 1))
    2022001
    >>> format_date_to_year_day(20220101)
    2022001
    """
    format_date = date
    if isinstance(date, datetime):
        format_date = int(date.strftime("%Y%j"))  # Get the year and day of the year for the current time
    elif isinstance(date, str):
        if len(date) == 7:
            format_date = int(date)
        elif len(date) == 8:
            format_date = int(datetime.strptime(date, "%Y%m%d").strftime("%Y%j"))
        else:
            print("date format error")
    elif isinstance(date, int):
        format_date = date
    else:
        print("date format error")
    return format_date


def convert_julian_regular_date(julian_date):
    # Convert Julian date to regular date
    year = int(str(julian_date)[:4])  # Extract year
    day_of_year = int(str(julian_date)[4:])  # Extract day of the year
    # Create date object
    date = datetime(year=year, month=1, day=1)  # Set to the first day of the year
    date += timedelta(days=day_of_year - 1)  # Add corresponding days
    print("Converted date:", date.strftime("%Y-%m-%d"))
    return date


def get_UTC_time(start_date, days=0):
    """
    Args:
        start_date: str, e.g. '2000-01-01 00:00:00'
        days: float, e.g. 7670.0625

    Returns: Returns UTC time
    """
    # Define start time
    start_time_utc = None
    if isinstance(start_date, datetime):
        start_time_utc = start_date.replace(tzinfo=timezone.utc)
    else:
        start_time_utc = datetime.strptime(
            start_date.astype(str).replace("000000000", "00"), "%Y-%m-%dT%H:%M:%S.%f"
        ).replace(tzinfo=timezone.utc)
    # Define time increment (7670.0625 days)
    days_increment = timedelta(days=days)
    # Calculate current time
    current_time = start_time_utc + days_increment
    return current_time


def get_Beijing_time_from_timezone(date_with_timezone, days=0):
    """
    Args:
    @ date_with_timezone: e.g. '2020-12-31T01:30:00.000000000'

    Returns: Returns Beijing time
    """
    # Convert UTC time to Beijing time
    utc_timezone = pytz.timezone("UTC")
    beijing_timezone = pytz.timezone("Asia/Shanghai")

    # Convert time string with timezone information to datetime object
    datetime_obj = datetime.strptime(
        date_with_timezone.astype(str).replace("000000000", "00"),
        "%Y-%m-%dT%H:%M:%S.%f",
    )
    utc_datetime = utc_timezone.localize(datetime_obj)
    beijing_datetime = utc_datetime.astimezone(beijing_timezone)
    beijing_datetime = beijing_datetime + timedelta(days=days)
    return beijing_datetime.astimezone(tz=None).replace(tzinfo=None)
    # Convert datetime object to regular time string
    # formatted_time = datetime_obj.strftime('%Y-%m-%d %H:%M:%S')


def get_Beijing_time(start_date, days):
    """
    Args:
        start_date: datetime
        days: float, e.g. 7670.0625
    Returns: Returns Beijing time
    """
    days_increment = timedelta(days=days)
    # Calculate current time
    current_time = start_date + days_increment
    current_time = current_time.astimezone(tz=None).replace(
        tzinfo=None
    )  # Convert time to local timezone
    # print(f"Current time: {current_time}")
    return current_time


def get_Beijing_time_from_UTC(start_date, days):
    """
    Args:
        start_date: str, e.g. '2000-01-01 00:00:00'
        days: float, e.g. 7670.0625
    Returns: Returns Beijing time
    """
    current_time = get_UTC_time(start_date, days)
    current_time = current_time.astimezone(tz=None).replace(
        tzinfo=None
    )  # Convert time to local timezone
    print(f"Current time: {current_time}")
    return current_time


def get_Beijing_time_from_UTC(utc_time):
    """
    Args:
        start_date: datetime, e.g. 2020-12-31 01:30:00+00:00
    Returns: Returns Beijing time
    """
    current_time = utc_time.astimezone(tz=None).replace(
        tzinfo=None
    )  # Convert time to local timezone
    print(f"Current time: {current_time}")
    return current_time


def get_closest_date_index(datetime_list, given_time):
    """
    Args:
        datetime_list: Collection of datetime objects
        given_time: Given time
    Returns:
    """

    # Find the index of the element in the collection closest to the given time
    closest_index = min(
        range(len(datetime_list)), key=lambda i: abs(datetime_list[i] - given_time)
    )
    return closest_index


def get_closest_hour_index(datetime_list, given_time):
    """
    Args:
        datetime_list: Collection of datetime objects
        given_time: Given time
    Returns:
    """
    # Find the index of the element in the collection closest to the given time
    closest_day_index = min(
        range(len(datetime_list)), key=lambda i: abs(datetime_list[i] - given_time)
    )
    closest_date = datetime_list[closest_day_index]
    hours = []
    for date in datetime_list:
        if (
            date.year == closest_date.year
            and date.month == closest_date.month
            and date.day == closest_date.day
        ):
            hours.append(date.hour)
    closest_hour_index = min(
        range(len(hours)), key=lambda i: abs(hours[i] - given_time.hour)
    )
    if given_time.tzinfo is not None:
        closest_hour_date = datetime(
            closest_date.year,
            closest_date.month,
            closest_date.day,
            hours[closest_hour_index],
            closest_date.minute,
            tzinfo=given_time.tzinfo,
        )
    else:
        closest_hour_date = datetime(
            closest_date.year,
            closest_date.month,
            closest_date.day,
            hours[closest_hour_index],
            closest_date.minute,
        )
    closest_index = datetime_list.index(closest_hour_date)
    return closest_index


def get_hour_of_year(date):
    """
    @description: Get the x-th hour of the year that the specified date belongs to
    @param: date: Date
    @return: The x-th hour of the year that the specified date belongs to
    @usage:
    """
    # Get the first day of the year
    start_of_year = datetime(date.year, 1, 1)
    # Calculate the time difference between the given date and the first day of the year
    time_difference = date - start_of_year
    # Convert time difference to hours
    hours_of_year = time_difference.total_seconds() / 3600
    # Add the hours of the current date
    hours_of_year += date.hour
    return int(hours_of_year)


def get_day_of_year_old(date):
    '''
    @description: Get the x-th day of the year that the specified date belongs to
    @param: date: Date
    @return: The x-th day of the year that the specified date belongs to
    @usage:
    '''
    start_of_year = datetime(date.year, 1, 1)
    # Calculate the time difference between the given date and the first day of the year
    time_difference = date - start_of_year
    # Convert time difference to hours
    days_of_year = int(time_difference.total_seconds() / (3600 * 24)) + 1
    # print(days_of_year)
    return days_of_year


def get_day_of_year(date):
    '''
    @description: Get the day of the year for the specified date
    @param {datetime}: date: Date
    @return: The day of the year for the specified date
    @usage:
    print(get_day_of_year(datetime(2021, 12, 31)))  # 365
    print(get_day_of_year(datetime(2020, 12, 31)))  # 366
    '''
    # Check if it's a string date
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d')
    return date.timetuple().tm_yday


def get_hour_of_month(date):
    '''
    @description: Get the x-th hour of the month that the specified date belongs to
    @param: date: Date
    @return: The x-th hour of the month that the specified date belongs to
    @usage:
    '''
    # Get the first day of the month
    start_of_year = datetime(date.year, date.month, 1)
    # Calculate the time difference between the given date and the first day of the month
    time_difference = date - start_of_year
    # Convert time difference to hours
    hours_of_year = time_difference.total_seconds() / 3600
    # Add the hours of the current date
    hours_of_year += date.hour
    return int(hours_of_year)


def get_days_in_month(month, year):
    """
    @description: Get the number of days in the specified month
    @param: month: Month
    @param: year: Year
    @return: Number of days in the month
    @usage:
    print(get_days_in_month(2, 2021))  # 29
    print(get_days_in_month(4, 2021))  # 30
    """
    return calendar.monthrange(year, month)[1]