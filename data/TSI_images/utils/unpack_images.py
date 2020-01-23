import os
import argparse as ap

def unpack_images(img_dir):
    files = os.listdir(img_dir)

    cam_hold = None
    
    for f in files:    
        year = f[:4]
        month = f[4:6]
        day = f[6:8]
        time = f.split('_')[0][8:]
        camera = f.split('_')[-1][:2]
        ext = f.split('_')[-1][-3:]
        
        if ext != 'jpg':
            continue
    
        else:
            pass

        fname = time+'.'+ext

        if camera==cam_hold:
            pass

        else:
            newpardir = os.path.abspath( os.path.join(os.path.curdir,camera,year,month,day) )
            os.system('mkdir -p {}'.format(newpardir))            
        
        oldfile = os.path.join(img_dir,f)
        newfile =  os.path.join(os.path.curdir,camera,year,month,day,fname)

        os.system('mv %s %s'%(oldfile,newfile))
        cam_hold = camera
        
if __name__=='__main__':    
    parser = ap.ArgumentParser(description='File to unpack images from')
    parser.add_argument('file', metavar='file', type=str, nargs=1)
    
    args = parser.parse_args()
    img_dir = args.file[0]
    unpack_images(img_dir)
