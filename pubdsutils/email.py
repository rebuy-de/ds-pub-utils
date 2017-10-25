import logging
import configparser
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

logger = logging.getLogger(__name__)


def set_email_server(config_ini):
    """
    Setting up an smtp client using INI file

    The INI file should have the following section:

    .. code-block:: ini

        [server]
        EMAIL_HOST = smtp.host.name.com
        EMAIL_HOST_USER = XXXXXXXXXXXXXXXXXXXX # For example Amazon user
        EMAIL_HOST_PASSWORD = YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
        EMAIL_PORT = 587
        EMAIL_TTLS = 1

    Parameters
    ----------
    config_ini : str
        Path to the config file

    Returns
    -------
    smtplib.SMTP
    """

    logger.info('Setting server email')
    with open(config_ini, 'r') as cfg_file:
        # config = yaml.load(cfg_file)
        config = configparser.ConfigParser()
        config.read_file(cfg_file)
    EMAIL_HOST = config['server']['EMAIL_HOST']
    EMAIL_HOST_USER = config['server']['EMAIL_HOST_USER']
    EMAIL_HOST_PASSWORD = config['server']['EMAIL_HOST_PASSWORD']
    EMAIL_PORT = config['server']['EMAIL_PORT']
    EMAIL_TTLS = config['server']['EMAIL_TTLS']
    if EMAIL_PORT == '':
        s = smtplib.SMTP(EMAIL_HOST)
    else:
        s = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
    if EMAIL_TTLS != '':
        s.starttls()
    if EMAIL_HOST_USER != '' and EMAIL_HOST_PASSWORD != '':
        s.login(EMAIL_HOST_USER, EMAIL_HOST_PASSWORD)
    return s


def send_df(df, config_ini, **kwargs):
    """
    Send a pandas.DataFrame per email as an attached CSV

    Parameters
    ----------
    estimate : pandas.DataFrame
        Data to be sent
    config_ini : str
        Path and filename of INI file. It should contain the section ``server``
        as described in :func:`set_email_server`. In addition, it has to
        contain the section ``content``:

        .. code-block:: ini

            [content]
            SUBJECT = Subject of the email (Generated on Y-m-d H:M:S
                                            will be appended)
            FROM = sent@from.here
            TO = send.to@foo.bar[, another@foo.bar]
            BODY = Body of the message. NO LINE BREAKS!
            FILENAME = foo.csv # note the .csv!
    **kwargs :
        passed to pandas.DataFrame.to_csv()
    """

    logger.info('Preparing and sending email')

    server = set_email_server(config_ini)

    config = configparser.ConfigParser()
    config.read(config_ini)

    sending_ts = datetime.now()
    sub = config['content']['SUBJECT'] + " / Generated on " + \
        sending_ts.strftime('%Y-%m-%d %H:%M:%S')

    msg = MIMEMultipart('alternative')
    msg['From'] = config['content']['FROM']
    msg['To'] = config['content']['TO']
    msg['Subject'] = sub

    body = config['content']['BODY']
    msg.attach(MIMEText(body, 'plain'))

    filename = config['content']['FILENAME']
    attachment = MIMEText(df.to_csv(**kwargs))
    attachment.add_header('Content-Disposition', 'attachment',
                          filename=filename)
    msg.attach(attachment)

    server.send_message(msg)
    server.quit()
