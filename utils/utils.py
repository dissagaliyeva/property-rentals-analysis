def change_price(row):
    """
    Check if the difference between predicted and true
    prices is bigger than $25. If so, store the newly-
    predicted price. Otherwise, keep the true value.

    Returns either the new predicted price or the actual price.

    Parameters
    ----------
    row : Pandas Series object
          Pandas Series object containing a data entry at specific
          index.

    Returns
    -------
          Newly predicted or actual prices.

    """

    # return the new price if price is off
    if row['off']:
        return row['predicted']

    # otherwise, return the actual price
    return row['price']