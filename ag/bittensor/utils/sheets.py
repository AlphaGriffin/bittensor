#!/usr/bin/python3
"""
CryptoSheets.
"""
import os, sys, time, datetime

import httplib2
from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage

import gspread

__author__ = "Eric Petersen @Ruckusist"
__copyright__ = "Copyright 2017, The Alpha Griffin Project"
__credits__ = ["Eric Petersen", "Shawn Wilson", "@alphagriffin"]
__license__ = "***"
__version__ = "0.0.2"
__maintainer__ = "Eric Petersen"
__email__ = "ruckusist@alphagriffin.com"
__status__ = "Beta"

class gHooks(object):
    """My interface to google."""

    def __init__(self, options):
        # build objects
        self.options = options

        # builds globals
        self.todays_report_filename = 'CryptoSheet {}'.format(
                datetime.datetime.now().strftime("%m-%d-%y"
                ))
        self.newfile = False

        # start init process
        self.gs = self.main()
        if self.options.notify_by_mail:
            self.notify()
        # self.spreadsheet = self.get_today()
        # self.spreadsheet.del_worksheet('Sheet1')

    def get_credentials(self):
        """Gets valid user credentials from storage.

        If nothing has been stored, or if the stored credentials are invalid,
        the OAuth2 flow is completed to obtain the new credentials.
        source: https://developers.google.com/sheets/api/quickstart/python
        Returns:
            Credentials, the obtained credential.
        """
        #### DONT EDIT.
        SCOPES = ['https://spreadsheets.google.com/feeds',
                  'https://www.googleapis.com/auth/drive']
        CLIENT_SECRET_FILE = 'client_secret.json'
        APPLICATION_NAME = 'reporter'
        ####
        home_dir = os.path.expanduser('~')
        credential_dir = os.path.join(home_dir, '.credentials')
        if not os.path.exists(credential_dir):
            os.makedirs(credential_dir)
        credential_path = os.path.join(credential_dir,
                                       'reporter_creds.json')

        store = Storage(credential_path)
        credentials = store.get()
        if not credentials or credentials.invalid:
            flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
            flow.user_agent = APPLICATION_NAME
            credentials = tools.run_flow(flow, store)
        return credentials

    def notify(self):
        # get from options. # FIXME
        if self.newfile:
            necessary_paries = [self.options.notify_by_mail,
                                'magilla422@gmail.com']
            for user in necessary_paries:
                # print("Sharing Sheet to Necessary Parties: {}".format(user))
                sheet.share(user, perm_type='user', role='writer')
            return True

    def get_today(self):
        try:
            sheet = self.gs.open(self.todays_report_filename)
            # print("Editing Todays Sheet.")
        except Exception as e:
            sheet = self.gs.create(self.todays_report_filename)
            # print("Opening Todays Sheet.")
            self.newfile = True
        return sheet

        #//////| Notify on Changes
        if newfile:
            self.notify()
        else:
            print("notifcations for today have been sent.")
        pass

    def get_template(self):
        template = None
        sheet = self.gs.open("TEMPLATE")
        return template

    def main(self):
        # login process
        creds = self.get_credentials()
        return gspread.authorize(creds)

def main():
    """Sheets updater Program."""
    start_time = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
    #//////| Build Objects
    # config = options.Options()
    P = printer.Printer()
    G = gHooks(config)
    P("AlphaGriffin.com | {}".format(start_time))

    #//////| Start Sequence
    P("Getting Google Creds.")
    creds = G.get_credentials()
    P("Authorizing Google Creds.")
    gc = gspread.authorize(creds)
    #//////| Get Spreadsheet
    todays_report_filename = 'CryptoSheet {}'.format(
            datetime.datetime.now().strftime("%m-%d-%y"))
    newfile = False
    try:
        sheet = gc.open(todays_report_filename)
        P("Editing Todays Sheet.")
    except Exception as e:
        sheet = gc.create(todays_report_filename)
        P("Opening Todays Sheet.")
        newfile = True

    #//////| Notify on Changes
    if newfile:
        # we dont need constant notifcations while testing.
        try:
            necessary_paries = ['eric.alphagriffin@gmail.com',
                                'magilla422@gmail.com']
            for user in necessary_paries:
                P("Sharing Sheet to Necessary Parties: {}".format(user))
                sheet.share(user, perm_type='user', role='writer')
        except:
            P("Insufficient Permission to share Spreadsheet. Fix this.")
    else:
        P("notifcations for today have been sent.")

    #//////| Open Worksheets
    P("Adding Worksheet Info to the Spreadsheet.")
    try:
        worksheet = sheet.add_worksheet(title="Info", rows="15", cols="10")
    except:
        worksheet = sheet.worksheet('Info')
    #//////| Start writing data to this worksheet.
    P("Populating this worksheet.")
    test1 = 'KinetechConcepts Info'
    worksheet.update_acell('A1', test1)
    worksheet.update_acell('A2', 'Last Updated: {}'.format(start_time))

    # DO MORE STUFF HERE.

    #//////| Finished |\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
    P("Finished Tests.")


if __name__ == '__main__':
    #try:
    import printer
    main()
    """
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("File: {1}\nError: {0}\nLine: {2}".format(
            exc_type, fname, exc_tb.tb_lineno
            ))
    """
