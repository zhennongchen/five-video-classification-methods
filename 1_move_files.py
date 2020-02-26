#!/usr/bin/env python

"""
After extracting the RAR, we run this to move all the files into
the appropriate train/test folders.

Should only run this file once!
"""
import os
import os.path
import shutil
import settings
import function_list as ff
cg = settings.Experiment()



def get_train_test_lists(main_path,version='01'):
    """
    Using one of the train/test files (01, 02, or 03), get the filename
    breakdowns we'll later use to move everything.
    """
    # Get our files based on version.
    test_file = os.path.join(main_path,'ucfTrainTestlist', 'testlist' + version + '.txt')
    train_file = os.path.join(main_path,'ucfTrainTestlist', 'trainlist' + version + '.txt')

    # Build the test list.
    with open(test_file) as fin:
        test_list = [row.strip() for row in list(fin)]

    # Build the train list. Extra step to remove the class index.
    with open(train_file) as fin:
        train_list = [row.strip() for row in list(fin)]
        train_list = [row.split(' ')[0] for row in train_list]
    print(len(train_list),len(test_list),train_list[0])
    # Set the groups in a dictionary.
    file_groups = {
        'train': train_list,
        'test': test_list
    }

    return file_groups

def copy_files(main_path,file_groups):
    """This assumes all of our files are currently in _this_ directory.
    So move them to the appropriate spot. Only needs to happen once.
    """
    # Do each of our groups.
    for group, videos in file_groups.items():
        
        # Do each of our videos.
        for video in videos:

            # Get the parts.
            parts = video.split(os.path.sep)
            classname = parts[0]
            filename = parts[1]
        

            # Check if this class exists.
            if not os.path.exists(os.path.join(main_path,group, classname)):
                print("Creating folder for %s/%s" % (group, classname))
                os.makedirs(os.path.join(main_path,group, classname))

            # Check if we have already moved this file, or at least that it
            # exists to move.
            if os.path.exists(os.path.join(main_path,group,classname,filename)):
                print(" find %s in the destination. Skipping." % (filename))
                continue

            if not os.path.exists(os.path.join(main_path,group,classname,filename)):
                #print(" can't find %s in the destination. copy it." % (filename))
                # copy the file
                original_file = os.path.join(main_path,'UCF_Videos',classname,filename)
                destination = os.path.join(main_path,group,classname,filename)
                shutil.copyfile(original_file,destination)
                

    
            # # Move it.
            # dest = os.path.join(group, classname, filename)
            # print("Moving %s to %s" % (filename, dest))
            # os.rename(filename, dest)
 

    print("Done copy")

def main():
    """
    Go through each of our train/test text files and move the videos
    to the right place.
    """
    main_path = os.path.join(cg.oct_main_dir,'UCF101')
    # Get the videos in groups so we can move them.
    
    group_lists = get_train_test_lists(main_path)

    # Move the files.
    copy_files(main_path,group_lists)

if __name__ == '__main__':
    main()
