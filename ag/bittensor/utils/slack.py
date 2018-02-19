#!/usr/bin/python3
"""
CryptoSlacker.
"""
import re, sys
# import os, sys, re, time, datetime
# from timeit import default_timer as timer
from slackclient import SlackClient
__author__ = "Eric Petersen @Ruckusist"
__copyright__ = "Copyright 2017, The Alpha Griffin Project"
__credits__ = ["Eric Petersen", "Shawn Wilson", "@alphagriffin"]
__license__ = "***"
__version__ = "0.0.2"
__maintainer__ = "Eric Petersen"
__email__ = "ruckusist@alphagriffin.com"
__status__ = "Beta"


# OAuth Access Token
slack_decider_api = 'xoxp-312902367441-313352940675-313577069573-a66360411bba466255f4b139079707db'
# Bot User OAuth Access Token
slack_decider_secret = 'xoxb-312805382672-cYAB78AW7a3Lz2MOcz6sHS7R'

class Slacker(object):
    """My interface to slack."""

    def __init__(self, options):
        self.options = options
        self.RTM_READ_DELAY = 1 # 1 second delay between reading from RTM
        self.EXAMPLE_COMMAND = ["do", "dont"]
        self.MENTION_REGEX = "^<@(|[WU].+?)>(.*)"
        self.slack_client = SlackClient(self.options.slack_decider_secret)

    def main(self):
        self.starterbot_id = None
        if self.slack_client.rtm_connect(with_team_state=False):
            print('Main Slacker Bot is Online.')
            self.Print('Decider Bot is coming online Now.')
            self.starterbot_id = self.slack_client.api_call("auth.test")["user_id"]
            try:
                while True:
                    command, channel, user = self.parse_bot_commands(self.slack_client.rtm_read())
                    if command:
                        self.handle_command(command, channel, user)
                    time.sleep(self.RTM_READ_DELAY)
            except KeyboardInterrupt:
                "Keyboard Shutdown Initiated."
                pass
            self.Print('Decider Bot is now offline.')
        return True

    def parse_bot_commands(self, slack_events):
        """
        Parses a list of events coming from the Slack RTM API to find bot commands.
        If a bot command is found, this function returns a tuple of command and channel.
        If its not found, then this function returns None, None.
        """
        for event in slack_events:
            user = 'Unknown User'
            try:
                user = event['user']
            except:
                pass
            finally:
                user_data = []
                if 'Unknown User' not in user:
                    try:
                        data = self.slack_client.api_call('users.info', user=user)
                        user = data['user']['profile']['display_name']
                        user_data.append(user)
                        name = data['user']['real_name']
                        user_data.append(name)
                        location = data['user']['tz']
                        user_data.append(location)
                    except:
                        user_data = ['','','']
                print("#! Event from @{}".format(user))
            for i, j in enumerate(event):
                print('{}: {}'.format(j, event[j]))
        if slack_events:
            print('#/ Events\n')
        for event in slack_events:
            if event["type"] == "message" and not "subtype" in event:
                user_id, message = self.parse_direct_mention(event["text"])
                if user_id == self.starterbot_id:
                    return message, event["channel"], user_data
        return None, None, None

    def parse_direct_mention(self, message_text):
        """
        Finds a direct mention (a mention that is at the beginning) in message text
        and returns the user ID which was mentioned. If there is no direct mention, returns None
        """
        matches = re.search(self.MENTION_REGEX, message_text)
        # the first group contains the username, the second group contains the remaining message
        return (matches.group(1), matches.group(2).strip()) if matches else (None, None)

    def handle_command(self, command, channel, user_data):
        """
        Executes bot command if the command is known
        """
        # Default response is help text for the user
        default_response = "@{} Only accepting *{}* Commands.".format(user_data[0], [x for x in self.EXAMPLE_COMMAND])

        # Finds and executes the given command, filling in response
        response = None
        # This is where you start to implement more commands!
        hit = False
        for commands in self.EXAMPLE_COMMAND:
            if command.lower().startswith(commands + ' '):
                response = "{}(@{}) from {} just said {}".format(user_data[1], user_data[0], user_data[2], str(command))

                # Sends the response back to the channel
                self.slack_client.api_call(
                    "chat.postMessage",
                    channel=channel,
                    text=response
                )
                hit = True
        if not hit:
            # Sends the response back to the channel
            self.slack_client.api_call(
                "chat.postMessage",
                channel=channel,
                text=default_response
            )

    def Print(self,  msg='test', channel='signals'):
        return self.slack_client.api_call(
                "chat.postMessage",
                channel=channel,
                text=msg
            )

if __name__ == '__main__':
    app = Slacker('config')
    if app.main():
        sys.exit('AlphaGriffin | 2018')
