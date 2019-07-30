from model import MusicTransformer
import params as par
from midi_processor import encode_midi, decode_midi

mt = MusicTransformer(
  embedding_dim=256, vocab_size=par.vocab_size, 
  num_layer=6, 
  max_seq=2048,
  dropout=0.1,
  debug=False,
  loader_path = 'result'
)

result0 = mt.generate(prior=[273., 369.,  56., 275., 185., 370.,  57., 272., 371., 55., 273.,
       370.,  52., 280., 364.,  53., 258., 371.,  50., 276., 185., 364.,
        57., 201., 196., 192., 189., 184., 183., 180., 181., 270., 178.,
       258., 368.,  65., 260., 185., 271., 371.,  69., 256., 371., 50.,], length=64)
# result1 = mt.generate(prior=[27, 186, 43, 213, 115, 131], length=512)
# result2 = mt.generate(prior=[1,3,4,5], length=10)

print('result0', len(result0), result0)

import midi_processor
midi0 = decode_midi(result0[0],file_path='result.midi')
# midi1 = decode_midi(result1[0],file_path='result.midi')
# midi2 = decode_midi(result2[0],file_path='result.midi')
print('midi0',midi0)