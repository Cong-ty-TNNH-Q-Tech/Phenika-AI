"""Run original CV/NLP training modules with local paths and bounded memory."""
import argparse
import ast
import json
import pathlib
import sys

ROOT=pathlib.Path(__file__).resolve().parents[2]
MODULES={'nodes':'cv/train_nodes.py','legend':'cv/train_legend.py',
         'landmark':'cv/train_landmark_resnet.py','edge':'cv/train_edge_resnet.py',
         'presence':'cv/train_presence.py','robot':'cv/train_robot_weather.py',
         'weather':'cv/train_weather_img.py','stairs':'cv/train_stairs_oneway.py',
         'oneway':'cv/train_oneway_dir.py','siamese':'cv/train_siamese.py',
         'nlp':'nlp/train_phobert_v3.py'}


class Localize(ast.NodeTransformer):
    def __init__(self,args):self.args=args

    def visit_Constant(self,node):
        value=node.value
        if isinstance(value,str):
            for old in ['/mnt/hdd2/qtech/Phenika-AI/Phenikaa_Campus_Courier_2026/Phenikaa_Campus_Courier_2026/delivery_public',
                        '/mnt/hdd2/qtech/Phenika-AI/v3data/delivery_public']:
                if value==old:return ast.copy_location(ast.Constant(str(self.args.data.resolve())),node)
            if value.startswith('/mnt/hdd2/qtech/Phenika-AI/solution/') and value.endswith('.pt'):
                # Includes f-string prefix used by stairs/oneway tasks.
                return ast.copy_location(ast.Constant(str(self.args.out.resolve()/pathlib.PurePosixPath(value).name)),node)
            if value.startswith('/mnt/hdd2/qtech/Phenika-AI/solution/') and value.endswith('/'):
                return ast.copy_location(ast.Constant(str(self.args.out.resolve())+'/'),node)
        return node

    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id=='DataLoader':
            for kw in node.keywords:
                if kw.arg=='num_workers':kw.value=ast.Constant(self.args.workers)
                elif kw.arg=='batch_size':kw.value=ast.Constant(self.args.batch)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='device' and node.args:
            node.args[0]=ast.Constant(self.args.device)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='from_pretrained':
            # Local cache is respected. Public model download occurs only when needed.
            pass
        return node


def main(args):
    import torch
    if args.device=='auto':args.device='cuda' if torch.cuda.is_available() else 'cpu'
    args.out.mkdir(parents=True,exist_ok=True)
    if args.task=='nlp' and not args.dry_run:
        try:import transformers
        except ImportError:raise RuntimeError('Install transformers and sentencepiece in the project environment first')
    script=ROOT/'solution'/MODULES[args.task]
    tree=ast.parse(script.read_text(encoding='utf8'))
    # Definition-only loading avoids automatic training, Windows worker recursion
    # and argparse from the old scripts. Original source/checkpoints stay intact.
    tree.body=[node for node in tree.body if not (isinstance(node,ast.If) and
               isinstance(node.test,ast.Compare) and isinstance(node.test.left,ast.Name) and
               node.test.left.id=='__name__')]
    # f-string model output is localized as a whole to preserve {task}.
    source=ast.unparse(tree)
    source=source.replace('/mnt/hdd2/qtech/Phenika-AI/solution/cv/',args.out.resolve().as_posix()+'/')
    tree=ast.parse(source)
    tree=Localize(args).visit(tree);ast.fix_missing_locations(tree)
    compiled=compile(tree,str(script),'exec')
    if args.dry_run:
        print(json.dumps({'task':args.task,'device':args.device,'data':str(args.data.resolve()),
                          'outputs':str(args.out.resolve()),'compiles':True}))
        return
    module_name='courier_local_'+args.task
    import types
    module=types.ModuleType(module_name);sys.modules[module_name]=module
    exec(compiled,module.__dict__)
    print('task',args.task,'device',args.device,'outputs',args.out,flush=True)
    if args.task=='robot':module.train_robot()
    elif args.task=='stairs':module.train_task('stairs')
    elif args.task=='nlp':module.run(epochs=args.epochs,bs=args.batch,accum=8,out=str(args.out/'phobert_v3.pt'))
    elif args.task=='nodes':module.run(epochs=args.epochs,n=2000,bs=args.batch)
    elif args.task=='legend':module.run(epochs=args.epochs,n=2000,bs=args.batch)
    elif args.task=='siamese':module.run(epochs=args.epochs,bs=args.batch)
    else:
        import inspect
        kwargs={'epochs':args.epochs,'bs':args.batch}
        signature=inspect.signature(module.run)
        module.run(**{k:v for k,v in kwargs.items() if k in signature.parameters})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('task',choices=list(MODULES))
    p.add_argument('--data',type=pathlib.Path,required=True)
    p.add_argument('--out',type=pathlib.Path,default=pathlib.Path('artifacts/local_checkpoints'))
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--dry-run',action='store_true')
    main(p.parse_args())
