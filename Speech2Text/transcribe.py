import argparse
import warnings
import csv
warnings.simplefilter('ignore')

from decoder import GreedyDecoder

from torch.autograd import Variable

from data.data_loader import SpectrogramParser
from model import DeepSpeech
import os
import json

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser.add_argument('--model-path', default='models/librispeech_final2.pth',
                    help='Path to model file created by training')
#parser.add_argument('--audio-path', default='/home/chesterguan/bigdataprojects/deepspeech.pytorch-master/data/Project-files/00.wav',
#                    help='Audio file to predict on')
parser.add_argument('--cuda', action="store_true", help='Use cuda to test model')
parser.add_argument('--decoder', default="greedy", choices=["greedy", "beam"], type=str, help="Decoder to use")
parser.add_argument('--offsets', dest='offsets', action='store_true', help='Returns time offset information')
beam_args = parser.add_argument_group("Beam Decode Options", "Configurations options for the CTC Beam Search decoder")
beam_args.add_argument('--top-paths', default=1, type=int, help='number of beams to return')
beam_args.add_argument('--beam-width', default=10, type=int, help='Beam width to use')
beam_args.add_argument('--lm-path', default=None, type=str,
                       help='Path to an (optional) kenlm language model for use with beam search (req\'d with trie)')
beam_args.add_argument('--alpha', default=0.8, type=float, help='Language model weight')
beam_args.add_argument('--beta', default=1, type=float, help='Language model word bonus (all words)')
beam_args.add_argument('--cutoff-top-n', default=40, type=int,
                       help='Cutoff number in pruning, only top cutoff_top_n characters with highest probs in '
                            'vocabulary will be used in beam search, default 40.')
beam_args.add_argument('--cutoff-prob', default=1.0, type=float,
                       help='Cutoff probability in pruning,default 1.0, no pruning.')
beam_args.add_argument('--lm-workers', default=1, type=int, help='Number of LM processes to use')
args = parser.parse_args()

def removerepeat(b):
    if b=="":
        return ""
    a=b[0]
    for i in b[1:]:
        if a[-1]!=i:
            a+=i
    return a
def decode_results(model, decoded_output, decoded_offsets):
    results = {
        "output": [],
        "_meta": {
            "acoustic_model": {
                "name": os.path.basename(args.model_path)
            },
            "language_model": {
                "name": os.path.basename(args.lm_path) if args.lm_path else None,
            },
            "decoder": {
                "lm": args.lm_path is not None,
                "alpha": args.alpha if args.lm_path is not None else None,
                "beta": args.beta if args.lm_path is not None else None,
                "type": args.decoder,
            }
        }
    }
    results['_meta']['acoustic_model'].update(DeepSpeech.get_meta(model))
    str=''
    print("len is : ",len(decoded_output))
    for b in range(len(decoded_output)):
        str2=''
        for pi in range(min(args.top_paths, len(decoded_output[b]))):
            result = {'transcription': decoded_output[b][pi]}
            #if(decoded_output[b][pi]!=" "):
            str2+=decoded_output[b][pi]
            if args.offsets:
                result['offsets'] = decoded_offsets[b][pi]
            results['output'].append(result)
        #str+=','
        #str+=removerepeat(str2)
        str+=str2   
    str=removerepeat(str)
    str=str.lower()
    print(str)
    #return results
    return str


if __name__ == '__main__':
    model = DeepSpeech.load_model(args.model_path, cuda=args.cuda)
    model.eval()

    labels = DeepSpeech.get_labels(model)
    audio_conf = DeepSpeech.get_audio_conf(model)

    if (args.decoder == "beam"):
        from decoder import BeamCTCDecoder
        decoder = BeamCTCDecoder(labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width, num_processes=args.lm_workers)
    else:
        decoder = GreedyDecoder(labels, blank_index=labels.index('_'))
    parser = SpectrogramParser(audio_conf, normalize=True)
    writer=csv.writer(open('./data/After_transcribelist.csv','w'))
    file_path='./data/Project-files6/'
    for(dirpath,dirnames,filenames) in os.walk(file_path):
        print('dirpath= '+dirpath)	
        for dirname in dirnames:
            print('dirname= '+dirname)
        for filename in filenames:
            name=filename.replace('.wav','')
            print(name)
            audio_path=os.path.join(dirpath,filename)
            spect = parser.parse_audio(audio_path).contiguous()
            spect = spect.view(1, 1, spect.size(0), spect.size(1))
            out = model(Variable(spect, volatile=True))
            out = out.transpose(0, 1)  # TxNxH
            decoded_output, decoded_offsets = decoder.decode(out.data)
            str=decode_results(model, decoded_output, decoded_offsets)
            writer.writerow([name,str])
            #print(json.dumps(decode_results(model, decoded_output, decoded_offsets)))
            
