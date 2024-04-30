import argparse
import utils
import MyData
import MyTrans

parse = argparse.ArgumentParser(description="machine translation using transformer")
parse.add_argument("--path_train", type=str, help="path to data train", default='data/train.json')
parse.add_argument("--path_valid", type=str, help="path to data valid", default='data/valid.json')
parse.add_argument("--path_test", type=str, help="path to data test", default='data/test.json')
parse.add_argument("--in_lang", type=str, help="input language", default='E')
parse.add_argument("--out_lang", type=str, help="output language", default='V')
parse.add_argument("--vocab_E", type=str, help="path Vocabulary of E", default='vocab/vocab_E.json')
parse.add_argument("--vocab_V", type=str, help="path Vocabulary of V", default='vocab/vocab_V.json')
parse.add_argument("--dmodel", type=int, help="Dimension of model", default=100)
parse.add_argument("--dembed", type=int, help="Dimension of embedding", default=100)
parse.add_argument("--d_ff", type=int, help="Dimension of feed-forward layer", default=400)
parse.add_argument("--head", type=int, help="Number of attention heads", default=4)
parse.add_argument("--active", type=str, help="Type of activation function", default="relu")
parse.add_argument("--layer", type=int, help="Number of layers", default=1)
parse.add_argument("--dropout", type=float, help="Dropout rate", default=0.1)
parse.add_argument("--eps", type=float, help="Epsilon value", default=1e-5)
parse.add_argument("--epoch", type= int, default= 10000, help= "epoch")
args = parse.parse_args()

if __name__ == "__main__":
    data_train = MyData.EV_Data(args.path_train, inp = args.in_lang, out = args.out_lang, E_vocab_path= args.vocab_E, V_vocab_path= args.vocab_V)
    model = MyTrans.Transformer(input_vocab_size= len(data_train.inp_vocab), output_vocab_size= len(data_train.out_vocab), dmodel = args.dmodel, dembed = args.dembed, 
                                d_ff= args.d_ff, head = args.head, active= args.active, layer= args.layer, dropout= args.dropout, eps = args.eps)     
    data_train = MyData.DataLoader(data_train, batch_size= 64, shuffle= True)
    if args.path_valid is not None:
        data_valid = MyData.EV_Data(args.path_valid, inp = args.in_lang, out = args.out_lang, E_vocab_path= args.vocab_E, V_vocab_path= args.vocab_V)
        data_valid = MyData.DataLoader(data_valid, batch_size= 64, shuffle= True)
        data_test = MyData.EV_Data(args.path_test, inp = args.in_lang, out = args.out_lang, E_vocab_path= args.vocab_E, V_vocab_path= args.vocab_V)
        data_test = MyData.DataLoader(data_test, batch_size= 64, shuffle= True)
    else:
        data_valid = None
        data_test = None
 
    optimizer = utils.optim.Adam(model.parameters())     
    utils.train(model, optimizer, args.epoch, data_train, data_valid, data_test)