from joblib import load

nrooms = int(input('Number of rooms (other than bedrooms): '))
total_area_sqft = int(input('Total area(sqft): '))
nbathrooms = int(input('Number of bathrooms: '))
nbedrooms = int(input('Number of bedrooms: '))

loaded_poly_converter = load('fpc.joblib')
loaded_mlmodel = load('fmodel.joblib')

user_input = [[nrooms, total_area_sqft, nbathrooms, nbedrooms]]
transformed_user_input = loaded_poly_converter.fit_transform(user_input)
predicted_output = loaded_mlmodel.predict(transformed_user_input)

print('The price would be nearly...')
print(round(predicted_output[0]), 'USD')
