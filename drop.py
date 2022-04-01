def drops(df):
    cleaned_df = df.copy()

    idx = cleaned_df[(cleaned_df['room_type'] == 'Shared room') &
                     (cleaned_df['property_type'] == 'Apartment') &
                     (cleaned_df['price'] > 50)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Shared room') &
                     (cleaned_df['property_type'] == 'Apartment') &
                     (cleaned_df['price'] < 32)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Private room') &
                     (cleaned_df['property_type'] == 'Apartment') &
                     (cleaned_df['price'] > 115)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Private room') &
                     (cleaned_df['property_type'] == 'Apartment') &
                     (cleaned_df['price'] < 50)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    # Entire apartment
    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'Apartment') &
                     (cleaned_df['bedrooms'] == 0) &
                     (cleaned_df['price'] < 100)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['bedrooms'] == 0) &
                     (cleaned_df['property_type'] == 'Apartment') &
                     (cleaned_df['price'] > 160)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['bedrooms'] == 1) &
                     (cleaned_df['property_type'] == 'Apartment') &
                     (cleaned_df['price'] < 130)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'Apartment') &
                     (cleaned_df['bedrooms'] == 1) &
                     (cleaned_df['price'] > 200)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'Apartment') &
                     (cleaned_df['bedrooms'] == 2) &
                     (cleaned_df['price'] < 150)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'Apartment') &
                     (cleaned_df['bedrooms'] == 2) &
                     (cleaned_df['price'] > 300)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'Apartment') &
                     (cleaned_df['bedrooms'] == 3) &
                     (cleaned_df['price'] < 240)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'Apartment') &
                     (cleaned_df['bedrooms'] == 4) &
                     (cleaned_df['price'] < 260)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    # house
    idx = cleaned_df[(cleaned_df['room_type'] == 'Shared room') &
                     (cleaned_df['property_type'] == 'House') &
                     (cleaned_df['price'] > 60)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Private room') &
                     (cleaned_df['property_type'] == 'House') &
                     (cleaned_df['price'] > 140)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    # entire home/apt
    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'House') &
                     (cleaned_df['bedrooms'] == 0) &
                     (cleaned_df['price'] < 80)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'House') &
                     (cleaned_df['bedrooms'] == 0) &
                     (cleaned_df['price'] > 150)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'House') &
                     (cleaned_df['bedrooms'] == 1) &
                     (cleaned_df['price'] > 200)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'House') &
                     (cleaned_df['bedrooms'] == 1) &
                     (cleaned_df['price'] < 120)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'House') &
                     (cleaned_df['bedrooms'] == 2) &
                     (cleaned_df['price'] < 140)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'House') &
                     (cleaned_df['bedrooms'] == 2) &
                     (cleaned_df['price'] > 350)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'House') &
                     (cleaned_df['bedrooms'] == 3) &
                     (cleaned_df['price'] < 240)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'House') &
                     (cleaned_df['bedrooms'] == 4) &
                     (cleaned_df['price'] < 270)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'House') &
                     (cleaned_df['bedrooms'] == 5) &
                     (cleaned_df['price'] < 300)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    # condominium
    idx = cleaned_df[(cleaned_df['room_type'] == 'Private room') &
                     (cleaned_df['property_type'] == 'Condominium') &
                     (cleaned_df['price'] > 120)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'Condominium') &
                     (cleaned_df['bedrooms'] == 0) &
                     (cleaned_df['price'] > 160)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'Condominium') &
                     (cleaned_df['bedrooms'] == 0) &
                     (cleaned_df['price'] < 100)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'Condominium') &
                     (cleaned_df['bedrooms'] == 1) &
                     (cleaned_df['price'] > 240)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'Condominium') &
                     (cleaned_df['bedrooms'] == 1) &
                     (cleaned_df['price'] < 125)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'Condominium') &
                     (cleaned_df['bedrooms'] == 2) &
                     (cleaned_df['price'] < 170)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'Condominium') &
                     (cleaned_df['bedrooms'] == 2) &
                     (cleaned_df['price'] > 300)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'Condominium') &
                     (cleaned_df['bedrooms'] == 3) &
                     (cleaned_df['price'] < 320)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'Condominium') &
                     (cleaned_df['bedrooms'] == 3) &
                     (cleaned_df['price'] > 450)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'Condominium') &
                     (cleaned_df['bedrooms'] == 4) &
                     (cleaned_df['price'] < 450)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'Condominium') &
                     (cleaned_df['bedrooms'] == 5)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    # Guest house
    idx = cleaned_df[(cleaned_df['room_type'] == 'Private room') &
                     (cleaned_df['property_type'] == 'Guest suite') &
                     (cleaned_df['price'] > 150)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'Guest suite') &
                     (cleaned_df['bedrooms'] == 0) &
                     (cleaned_df['price'] > 160)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'Guest suite') &
                     (cleaned_df['bedrooms'] == 1) &
                     (cleaned_df['price'] < 100)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'Guest suite') &
                     (cleaned_df['bedrooms'] == 1) &
                     (cleaned_df['price'] > 250)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'Guest suite') &
                     (cleaned_df['bedrooms'] == 2) &
                     (cleaned_df['price'] < 150)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Entire home/apt') &
                     (cleaned_df['property_type'] == 'Guest suite') &
                     (cleaned_df['bedrooms'] == 2) &
                     (cleaned_df['price'] > 300)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    # boutique hotel
    idx = cleaned_df[(cleaned_df['room_type'] == 'Private room') &
                     (cleaned_df['property_type'] == 'Boutique hotel') &
                     (cleaned_df['price'] < 160)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Private room') &
                     (cleaned_df['property_type'] == 'Boutique hotel') &
                     (cleaned_df['price'] > 200)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Hotel room') &
                     (cleaned_df['property_type'] == 'Boutique hotel') &
                     (cleaned_df['price'] > 190)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    # hotel
    idx = cleaned_df[(cleaned_df['room_type'] == 'Private room') &
                     (cleaned_df['property_type'] == 'Hotel') &
                     (cleaned_df['price'] < 150)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    idx = cleaned_df[(cleaned_df['room_type'] == 'Private room') &
                     (cleaned_df['property_type'] == 'Hotel') &
                     (cleaned_df['price'] > 250)].index
    cleaned_df.drop(index=idx, axis='rows', inplace=True)

    cleaned_df.drop(index=[2662, 6310], axis='rows', inplace=True)

    return cleaned_df
