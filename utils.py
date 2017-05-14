import os

def soundNotification():
    """Ring a bell as notification"""
    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % ( 0.6, 330))



def mailNotification():
    pass
