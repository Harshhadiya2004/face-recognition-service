class MockMilvus:

    def get_collection_stats(self):
        return {"num_entities": 10, "name": "face_collection"}

    def user_exists(self, user_id, org):
        return False

    def check_face_duplicate(self, **kwargs):
        return None

    def insert_embedding(self, user_id, emb, org):
        return 1

    def search_similar(self, emb, org, threshold):
        return [("USER101", 0.91)]

    def update_embedding(self, user_id, emb, org):
        return 1

    def delete_user(self, user_id, org):
        return 1
