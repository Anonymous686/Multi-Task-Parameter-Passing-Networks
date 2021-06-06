import torch
import collections

sl=0
nl=0
nl2=0
nl3=0
dl=0
el=0
rl=0
kl=0
tl=0
al=0
cl=0
popular_offsets=collections.defaultdict(int)
batch_number=0

def segment_semantic_loss(output,target,mask):
    global sl
    sl = torch.nn.functional.cross_entropy(output.float(),target.long().squeeze(dim=1),ignore_index=0,reduction='mean')
    return sl

def normal_loss(output,target,mask):
    global nl
    nl= rotate_loss(output,target,mask,normal_loss_base)
    return nl
    
def normal_loss_simple(output,target,mask):
    global nl
    out = torch.nn.functional.l1_loss(output,target,reduction='none')
    out *=mask.float()
    nl = out.mean()
    return nl
   
def rotate_loss(output,target,mask,loss_name):
    global popular_offsets
    target=target[:,:,1:-1,1:-1].float()
    mask = mask[:,:,1:-1,1:-1].float()
    output=output.float()
    val1 = loss = loss_name(output[:,:,1:-1,1:-1],target,mask)
    
    val2 = loss_name(output[:,:,0:-2,1:-1],target,mask)
    loss = torch.min(loss,val2)
    val3 = loss_name(output[:,:,1:-1,0:-2],target,mask)
    loss = torch.min(loss,val3)
    val4 = loss_name(output[:,:,2:,1:-1],target,mask)
    loss = torch.min(loss,val4)
    val5 = loss_name(output[:,:,1:-1,2:],target,mask)
    loss = torch.min(loss,val5)
    val6 = loss_name(output[:,:,0:-2,0:-2],target,mask)
    loss = torch.min(loss,val6)
    val7 = loss_name(output[:,:,2:,2:],target,mask)
    loss = torch.min(loss,val7)
    val8 = loss_name(output[:,:,0:-2,2:],target,mask)
    loss = torch.min(loss,val8)
    val9 = loss_name(output[:,:,2:,0:-2],target,mask)
    loss = torch.min(loss,val9)
    
    #lst = [val1,val2,val3,val4,val5,val6,val7,val8,val9]

    #print(loss.size())
    loss=loss.mean()
    #print(loss)
    return loss


def normal_loss_base(output,target,mask):
    out = torch.nn.functional.l1_loss(output,target,reduction='none')
    out *=mask
    out = out.mean(dim=(1,2,3))
    return out

def normal2_loss(output,target,mask):
    global nl3
    diff = output.float() - target.float()
    out = torch.abs(diff)
    out = out*mask.float()
    nl3 = out.mean()
    return nl3

def depth_loss_simple(output,target,mask):
    global dl
    out = torch.nn.functional.l1_loss(output,target,reduction='none')
    out *=mask.float()
    dl = out.mean()
    return dl

def depth_loss(output,target,mask):
    global dl
    dl = rotate_loss(output,target,mask,depth_loss_base)
    return dl

def depth_loss_base(output,target,mask):
    out = torch.nn.functional.l1_loss(output,target,reduction='none')
    out *=mask.float()
    out = out.mean(dim=(1,2,3))
    return out

def edge_loss_simple(output,target,mask):
    global el
    
    out = torch.nn.functional.l1_loss(output,target,reduction='none')
    out *=mask
    el = out.mean()
    return el

def reshade_loss(output,target,mask):
    global rl
    out = torch.nn.functional.l1_loss(output,target,reduction='none')
    out *=mask
    rl = out.mean()
    return rl

def keypoints2d_loss(output,target,mask):
    global kl
    kl = torch.nn.functional.l1_loss(output,target)
    return kl

def edge2d_loss(output,target,mask):
    global tl
    tl = torch.nn.functional.l1_loss(output,target)
    return tl

def auto_loss(output,target,mask):
    global al
    al = torch.nn.functional.l1_loss(output,target)
    return al

def pc_loss(output,target,mask):
    global cl
    out = torch.nn.functional.l1_loss(output,target,reduction='none')
    out *=mask
    cl = out.mean()
    return cl

def edge_loss(output,target,mask):
    global el
    out = torch.nn.functional.l1_loss(output,target,reduction='none')
    out *=mask
    el = out.mean()
    return el


def get_taskonomy_loss(losses):
    def taskonomy_loss(output,target):
        if 'mask' in target:
            mask = target['mask']
        else:
            mask=None

        sum_loss=None
        num=0
        for n,t in target.items():
            if n in losses:
                o = output[n].float()
                this_loss = losses[n](o,t,mask)
                num+=1
                if sum_loss:
                    sum_loss = sum_loss+ this_loss
                else:
                    sum_loss = this_loss
        
        return sum_loss#/num # should not take average when using xception_taskonomy_new
    return taskonomy_loss


def get_losses_and_tasks(args):
    task_str = args.tasks
    losses = {}
    criteria = {}
    taskonomy_tasks = []
        
    if 's' in task_str:
        losses['segment_semantic'] = segment_semantic_loss
        criteria['ss_l']=lambda x,y : sl
        taskonomy_tasks.append('segment_semantic')
    if 'd' in task_str:
        if not args.rotate_loss:
            losses['depth_zbuffer'] = depth_loss_simple
        else:            
            # print('got rotate loss')
            losses['depth_zbuffer'] = depth_loss
        criteria['depth_l']=lambda x,y : dl
        taskonomy_tasks.append('depth_zbuffer')

    if 'n' in task_str:
        if not args.rotate_loss:
            losses['normal']=normal_loss_simple
        else:
            # print('got rotate loss')
            losses['normal']=normal_loss
        criteria['norm_l']=lambda x,y : nl
        taskonomy_tasks.append('normal')
    if 'N' in task_str:
        losses['normal2']=normal2_loss
        criteria['norm2']=lambda x,y : nl3
        taskonomy_tasks.append('normal2')
    if 'k' in task_str:
        losses['keypoints2d']=keypoints2d_loss
        criteria['key_l']=lambda x,y : kl
        taskonomy_tasks.append('keypoints2d')
    if 'e' in task_str:
        if not args.rotate_loss:
            losses['edge_occlusion'] = edge_loss_simple
        else:            
            # print('got rotate loss')
            losses['edge_occlusion'] = edge_loss
        criteria['edge_l']=lambda x,y : el
        taskonomy_tasks.append('edge_occlusion')
    if 'r' in task_str:
        losses['reshading']=reshade_loss
        criteria['shade_l']=lambda x,y : rl
        taskonomy_tasks.append('reshading')
    if 't' in task_str:
        losses['edge_texture']=edge2d_loss
        criteria['edge2d_l']=lambda x,y : tl
        taskonomy_tasks.append('edge_texture')
    if 'a' in task_str:
        losses['rgb']=auto_loss
        criteria['rgb_l']=lambda x,y : al
        taskonomy_tasks.append('rgb')
    if 'c' in task_str:
        losses['principal_curvature']=pc_loss
        criteria['pc_l']=lambda x,y : cl
        taskonomy_tasks.append('principal_curvature')

    if args.task_weights:
        weights=[float(x) for x in args.task_weights.split(',')]

        for l,w,c in zip(losses.items(),weights,criteria.items()):
            losses[l[0]]=lambda x,y,z,l=l[1],w=w:l(x,y,z)*w
            criteria[c[0]]=lambda x,y,c=c[1],w=w:c(x,y)*w

    taskonomy_loss = get_taskonomy_loss(losses)
    return taskonomy_loss, losses, criteria, taskonomy_tasks


def get_losses_and_tasks_multicase(args):
    losses = {}
    criteria = {}
    taskonomy_tasks = []

    for building_idx, task_idx in args.building_task_mappings:
        task_str = args.tasks[task_idx]

        if 's' in task_str:
            losses[args.building_names[building_idx] + '/' + 'segment_semantic'] = segment_semantic_loss
            criteria[str(building_idx) + '/' + 'ss_l'] = \
                lambda loss_d, idx: loss_d[args.building_names[idx] + '/' + 'segment_semantic']
            taskonomy_tasks.append(args.building_names[building_idx] + '/' + 'segment_semantic')
        if 'd' in task_str:
            if not args.rotate_loss:
                losses[args.building_names[building_idx] + '/' + 'depth_zbuffer'] = depth_loss_simple
            else:
                # print('got rotate loss')
                losses[args.building_names[building_idx] + '/' + 'depth_zbuffer'] = depth_loss
            criteria[str(building_idx) + '/' + 'depth_l'] = \
                lambda loss_d, idx: loss_d[args.building_names[idx] + '/' + 'depth_zbuffer']
            taskonomy_tasks.append(args.building_names[building_idx] + '/' + 'depth_zbuffer')

        if 'n' in task_str:
            if not args.rotate_loss:
                losses[args.building_names[building_idx] + '/' + 'normal'] = normal_loss_simple
            else:
                # print('got rotate loss')
                losses[args.building_names[building_idx] + '/' + 'normal'] = normal_loss
            criteria[str(building_idx) + '/' + 'norm_l'] = \
                lambda loss_d, idx: loss_d[args.building_names[idx] + '/' + 'normal']
            taskonomy_tasks.append(args.building_names[building_idx] + '/' + 'normal')
        if 'N' in task_str:
            losses[args.building_names[building_idx] + '/' + 'normal2'] = normal2_loss
            criteria[str(building_idx) + '/' + 'norm2'] = \
                lambda loss_d, idx: loss_d[args.building_names[idx] + '/' + 'normal2']
            taskonomy_tasks.append(args.building_names[building_idx] + '/' + 'normal2')
        if 'k' in task_str:
            losses[args.building_names[building_idx] + '/' + 'keypoints2d'] = keypoints2d_loss
            criteria[str(building_idx) + '/' + 'key_l'] = \
                lambda loss_d, idx: loss_d[args.building_names[idx] + '/' + 'keypoints2d']
            taskonomy_tasks.append(args.building_names[building_idx] + '/' + 'keypoints2d')
        if 'e' in task_str:
            if not args.rotate_loss:
                losses[args.building_names[building_idx] + '/' + 'edge_occlusion'] = edge_loss_simple
            else:
                # print('got rotate loss')
                losses[args.building_names[building_idx] + '/' + 'edge_occlusion'] = edge_loss
            criteria[str(building_idx) + '/' + 'edge_l'] = \
                lambda loss_d, idx: loss_d[args.building_names[idx] + '/' + 'edge_occlusion']
            taskonomy_tasks.append(args.building_names[building_idx] + '/' + 'edge_occlusion')
        if 'r' in task_str:
            losses[args.building_names[building_idx] + '/' + 'reshading'] = reshade_loss
            criteria[str(building_idx) + '/' 'shade_l'] = \
                lambda loss_d, idx: loss_d[args.building_names[idx] + '/' + 'reshading']
            taskonomy_tasks.append(args.building_names[building_idx] + '/' + 'reshading')
        if 't' in task_str:
            losses[args.building_names[building_idx] + '/' + 'edge_texture'] = edge2d_loss
            criteria[str(building_idx) + '/' + 'edge2d_l'] = \
                lambda loss_d, idx: loss_d[args.building_names[idx] + '/' + 'edge_texture']
            taskonomy_tasks.append(args.building_names[building_idx] + '/' + 'edge_texture')
        if 'a' in task_str:
            losses[args.building_names[building_idx] + '/' + 'rgb'] = auto_loss
            criteria[str(building_idx) + '/' + 'rgb_l'] = \
                lambda loss_d, idx: loss_d[args.building_names[idx] + '/' + 'rgb']
            taskonomy_tasks.append(args.building_names[building_idx] + '/' + 'rgb')
        if 'c' in task_str:
            losses[args.building_names[building_idx] + '/' + 'principal_curvature'] = pc_loss
            criteria[str(building_idx) + '/' + 'pc_l'] = \
                lambda loss_d, idx: loss_d[args.building_names[idx] + '/' + 'principal_curvature']
            taskonomy_tasks.append(args.building_names[building_idx] + '/' + 'principal_curvature')

        if args.task_weights:
            weights = [float(x) for x in args.task_weights.split(',')]

            for l, w, c in zip(losses.items(), weights, criteria.items()):
                losses[l[0]] = lambda x, y, z, l=l[1], w=w: l(x, y, z) * w
                criteria[c[0]] = lambda x, y, c=c[1], w=w: c(x, y) * w

    taskonomy_loss = get_taskonomy_loss_multicase(losses)
    return taskonomy_loss, losses, criteria, taskonomy_tasks


def get_taskonomy_loss_multicase(losses):
    def taskonomy_loss_multicase(output_dict, target_dict):
        sum_loss = None
        num = 0
        loss_d = {}
        for real_task in output_dict:
            temp_mask = real_task.split('/')[0] + '/' + 'mask'
            if temp_mask in target_dict:
                mask = target_dict[temp_mask]
            else:
                mask = None
            if real_task in losses:
                o = output_dict[real_task].float()
                this_loss = losses[real_task](o, target_dict[real_task], mask)
                loss_d[real_task] = this_loss
                num += 1
                if sum_loss:
                    sum_loss = sum_loss + this_loss
                else:
                    sum_loss = this_loss

        return sum_loss, loss_d  # /num # should not take average when using xception_taskonomy_new

    return taskonomy_loss_multicase