from upstash_vector import Index

index = Index(url="https://optimal-spaniel-5472-us1-vector.upstash.io", token="ABgFMG9wdGltYWwtc3BhbmllbC01NDcyLXVzMWFkbWluTmpGalptTm1ZMkl0WlRCaVlpMDBOVEl6TFdGa1pUTXRZekV4WmpreFltVmxOVFUy")

index.upsert(
  vectors=[
      ("id1", "Enter data as string", {"metadata_field": "metadata_value"}),
  ]
)

index.query(
  data="Enter data as string",
  top_k=1,
  include_vectors=True,
  include_metadata=True
)

