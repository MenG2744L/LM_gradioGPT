from fastapi import FastAPI

app = FastAPI()


@app.post("/infer/{p1}/{p2}")
def infer(p1, p2):
    # todo something...
    pass
