import os

def filterList(inp,pattern,mode='include'):
    out = []
    
    #print inp, 'pattern=', pattern
    if mode =='include':
        for el in inp:
            if el.find(pattern) != -1:
                out.append(el)
    if mode == 'avoid':
        for el in inp:
            if el.find(pattern) == -1:
                out.append(el)

    return out      




def getFilelistFromDir(folder,pattern,avoidpattern=None):
    """Returns a list of files from a given source folder. 
    Filenames must match the list of patterns and must not include
    the optional list of avoidpatterns 

    :param folder: root folder where files are located
    :type folder: string
    :param pattern: string patterns. each filename in the output filename list matches all patterns 
    :param avoidpattern: string patterns. each filename in the output list does not match any of the avoidpatterns
    :type pattern: list of strings
    :type avoidpattern: list of strings
    :rtype: list of strings
    """
     

    allfiles = os.listdir(folder)
    selectedfiles = []

    pt = my_toList(pattern) #convert pattern string to list
    flist = allfiles    
    for patel in pt:
        flist =  filterList(flist, patel)
    

    if avoidpattern:
        av = my_toList(avoidpattern)
        for patel in av:
            flist = filterList(flist, patel, mode = 'avoid')
    return flist        


def my_toList(pattern):
    if type(pattern) is str: # if only one pattern is available
        return [pattern]
    else:
        return pattern  
