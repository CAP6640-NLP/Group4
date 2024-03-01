
# Tutorial on how to use this pretrained model on an IMDB data set taken from here "https://pytorch.org/text/stable/tutorials/t5_demo.html#sphx-glr-tutorials-t5-demo-py"

# This model shows our ultimate goal with the transformer we coded from scratch, where we would take a review as input and the transformer would output if is was a positive or negative review

from torchtext.models import T5_BASE_GENERATION
from functools import partial
from torch.utils.data import DataLoader
from torchtext.prototype.generate import GenerationUtils
from torchtext.datasets import IMDB
from torchtext.models import T5Transform
import torchdata.datapipes as dp

t5_base = T5_BASE_GENERATION
padding_idx = 0
eos_idx = 1
max_seq_len = 512
transform = t5_base.transform()
model = t5_base.get_model()
model.eval()

def apply_prefix(task, x):
    return f"{task}: " + x[0], x[1]

def process_labels(labels, x):
    return x[1], labels[str(x[0])]

sequence_generator = GenerationUtils(model)

imdb_batch_size = 100
imdb_datapipe = IMDB(split="test")
task = "sst2 sentence"
labels = {"1": "negative", "2": "positive"}

imdb_datapipe = imdb_datapipe.map(partial(process_labels, labels))
imdb_datapipe = imdb_datapipe.map(partial(apply_prefix, task))

imdb_datapipe = imdb_datapipe.batch(imdb_batch_size)
imdb_datapipe = imdb_datapipe.rows2columnar(["review", "sentiment"])
imdb_dataloader = DataLoader(imdb_datapipe, batch_size=None)

batch = next(iter(imdb_dataloader))
input_text = batch["review"]
target = batch["sentiment"]
beam_size = 1

model_input = transform(input_text)
model_output = sequence_generator.generate(model_input, num_beams=beam_size)
output_text = transform.decode(model_output.tolist())
correct_guess = 0

print(f"Here is an example of the transormer predicting the sentiment of a review")
print(f"Input text: {input_text[0]}\n")
print(f"Transformers Prediction: {output_text[0]} Actual Sentiment: {target[0]}")

for i in range(imdb_batch_size):
    if target[i] == output_text[i]:
        correct_guess += 1

print(f"The pretrained transfromer gets {(correct_guess/imdb_batch_size) * 100}% of its predictions right on the IMBD data set")