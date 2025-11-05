from openai import OpenAI
client = OpenAI()

batch = client.batches.retrieve("batch_690a150d8e5c81909be19ff9ff46feed")
print(batch)
