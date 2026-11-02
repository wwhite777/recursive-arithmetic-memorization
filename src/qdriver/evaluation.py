import math
from collections import Counter
import torch
from reasoning import arithmetic
from .training import tensors
from .common import require

def histogram(rows):
    return Counter((len(r["operands"]),len(str(r["result"]))) for r in rows)

def evaluate_split(trainer,rows,batch_size,device,expected_counts=None):
    groups={key:[] for key in ((n,w) for n in range(3,9) for w in (1,2,3))}
    for row in rows:
        require(row['split'] in ('train','id_dev'),'attempted forbidden split')
        key=(len(row['operands']),len(str(row['result'])));require(key in groups,'unsupported stratum')
        require(sum(row['target_mask'])==key[0]-1,'operator denominator inconsistent with operand count');groups[key].append(row)
    actual=histogram(rows)
    if expected_counts is not None:require(actual==Counter(expected_counts),'fixed stratum count mismatch')
    cells=[];weighted_losses=[];strict=semantic=errors=operator_total=0
    for (n,width),current in groups.items():
        count=len(current);tokens=count*(n-1);correct=valid=invalid=0;loss_terms=[]
        for first in range(0,count,batch_size):
            chunk=current[first:first+batch_size];batch={k:v.to(device) for k,v in tensors(chunk).items()}
            output=trainer.evaluate_batch(batch);loss=output['loss'];require(type(loss) in (int,float) and math.isfinite(loss) and loss>=0,'nonfinite evaluation loss')
            require(type(output['strict_operator_correct']) is int and 0<=output['strict_operator_correct']<=len(chunk),'strict count invalid')
            prediction=output['predictions'].tolist();require(len(prediction)==len(chunk) and all(len(p)==7 for p in prediction),'prediction count/shape')
            correct+=output['strict_operator_correct'];loss_terms.append(loss*len(chunk)*(n-1))
            for row,p in zip(chunk,prediction):
                operators=[op for op,flag in zip(p,row['target_mask']) if flag]
                try:valid+=int(arithmetic.evaluate_rpn(row['structure'],row['operands'],operators)==row['result'])
                except ValueError:invalid+=1
        require(correct<=valid and valid+invalid<=count,'strict/semantic/error inconsistency')
        loss_sum=math.fsum(loss_terms);weighted_losses.append(loss_sum);strict+=correct;semantic+=valid;errors+=invalid;operator_total+=tokens
        cells.append({'operand_count':n,'result_width':width,'present':count>0,'total':count,'strict_correct':correct,'semantic_correct':valid,
            'error_count':invalid,'valid_operator_token_total':tokens,'mean_loss':loss_sum/tokens if tokens else None,
            'strict_accuracy':correct/count if count else None,'semantic_accuracy':valid/count if count else None})
    require(rows and operator_total>0 and strict<=len(rows) and semantic+errors<=len(rows),'empty/inconsistent eval totals')
    return {'total':len(rows),'strict_correct':strict,'semantic_correct':semantic,'semantic_error_count':errors,
        'valid_operator_token_total':operator_total,'mean_loss':math.fsum(weighted_losses)/operator_total,
        'strict_accuracy':strict/len(rows),'semantic_accuracy':semantic/len(rows),'strata':cells,
        'scope':'Same actual final EMA trainer, full ACT8, evaluation mode; canonical strict and secondary semantic counts'}

