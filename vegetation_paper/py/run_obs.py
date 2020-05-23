import os
import time
import schedule



def take_scan(cnt):
    """
    Take a VNIR scan.

    cnt - the current scan number
    """

    # -- set file name
    fname = "veg_{0:05}".format(cnt[0])

    # -- define commands
    sop_cmd  = os.path.join("..","msv",
                            "MSV.Specials.NYU.PeripheralControl.exe -s1")
    scan_cmd = "MSV.Measure.exe " + \
               "-c ..\input\CameraConfig_050216_test0.xml "  + \
               "-p ..\input\PTUConfig_050216_test0.xml "  + \
               "-s {0} ".format(fname) + \
               "-d ..\data\ "
    scl_cmd  = os.path.join("..","msv",
                            "MSV.Specials.NYU.PeripheralControl.exe -s0")
    
    # -- open shutter
    sop = os.system(sop_cmd)

    # -- check if shutter command worked
    if sop!=0:
        fopen = open(os.path.join("..","output",
                                  "log_sop_{0:05}.txt".format(cnt[0])))
        fopen.write("shutter didn't open!!!")
        fopen.close()

    # -- take scan
    cwd  = os.getcwd()
    cd0  = os.chdir(os.path.join("..","msv"))
    scan = os.system(scan_cmd)
    cd1  = os.chdir(os.path.join(cwd))

    # -- close shutter
    scl = os.system(scl_cmd)

    # -- check if shutter command worked
    if scl!=0:
        fopen = open(os.path.join("..","output",
                                  "log_scl_{0:05}.txt".format(cnt[0])))
        fopen.write("shutter didn't close!!!")
        fopen.close()

    return


    
if __name__=="__main__":

    # -- schedule
    times = ["{0:02}:{1:02}".format(i,j) for i in range(8,18) for j in
             [0,10,15,45]] + ["18:00"]

    for cnt,itime in enumerate(times):
        schedule.every().day.at(itime).do(take_scan,cnt)

    while True:
        schedule.run_pending()
        time.sleep(1)
