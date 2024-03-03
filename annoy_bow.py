from flask import Flask, request, jsonify
from annoy import AnnoyIndex

app = Flask(__name__)

# we Load the Annoy database
index_file_path = 'bag_ofword.ann'
vector_dimension = 15134
annoy_index = AnnoyIndex(vector_dimension,'angular')
annoy_index.load(index_file_path)

@app.route('/reco', methods=['POST']) # This route is used to get recommendations
def reco():
    vector = request.json['vector'] # Get the vector from the request
    closest_indices = annoy_index.get_nns_by_vector(vector, 5) # Get the 5 closest elements indices
    reco = [closest_indices[0], closest_indices[1],closest_indices[2],closest_indices[3],closest_indices[4]]
    return jsonify(reco) # Return the reco as a JSON

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005) # Run the server on port 5000 and make it accessible externally