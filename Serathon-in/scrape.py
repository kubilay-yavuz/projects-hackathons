import requests
import zipfile
from io import BytesIO
import datetime
import os

date = datetime.date.today()
datedelta = datetime.timedelta(days=1)
while date.year > 2000:
    if date.weekday() > 4:
        date = date - datedelta
        continue
    filename_ = date.strftime("%Y%m%d")
    if os.path.exists("files/{0}.csv".format(filename_)) or os.path.exists("files/{0}.xls".format(filename_)):
        print("exists")
        date = date - datedelta
        continue

    datestr1 = date.strftime("%Y-%m-%d")
    print(datestr1)
    datestr2 = date.strftime("%#d.%#m.%Y")
    data={
        'ctl17_TSM': ';;System.Web.Extensions, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35:tr-TR:b7585254-495e-4311-9545-1f910247aca5:ea597d4b:b25378d2;;Telerik.Sitefinity, Version=5.4.4010.0, Culture=neutral, PublicKeyToken=b28c218413bdf563:tr-TR:b32eae45-c0e9-40fb-a6c7-4789a371d8b6:3b9a1b05;Telerik.Sitefinity.Search.Impl, Version=5.4.4010.0, Culture=neutral, PublicKeyToken=b28c218413bdf563:tr-TR:b69a937b-4441-4700-bea2-119dd5c3612a:7561727d;Telerik.Web.UI, Version=2012.3.1308.40, Culture=neutral, PublicKeyToken=121fae78165ba3d4:tr-TR:ed0b2505-4ecf-45ea-bd67-f51c9ccd1855:16e4e7cd:f7645509:8674cba1:7c926187:b7778d6c:c08e9f8a:a51ee93e:59462f1',
        'ctl18_TSSM': ';Telerik.Sitefinity.Resources, Version=5.4.4010.0, Culture=neutral, PublicKeyToken=null:tr:6264c692-5b24-4cea-9f2c-9f5d95f4d9d9:d271177c:dff30785;Telerik.Web.UI, Version=2012.3.1308.40, Culture=neutral, PublicKeyToken=121fae78165ba3d4:tr:ed0b2505-4ecf-45ea-bd67-f51c9ccd1855:aac1aeb7:c73cf106:c86a4a06:4c651af2',
        '__EVENTTARGET': '',
        '__EVENTARGUMENT': '',
        '__VIEWSTATE': '/wEPDwULLTIwNTUxMTMxMzJkGAEFHl9fQ29udHJvbHNSZXF1aXJlUG9zdEJhY2tLZXlfXxYUBTBjdGwwMCRUZXh0Q29udGVudCRDMDAxJHJkcEhpc3NlU2VuZXRsZXJpUGl5YXNhc2kFOWN0bDAwJFRleHRDb250ZW50JEMwMDEkcmRwSGlzc2VTZW5ldGxlcmlQaXlhc2FzaSRjYWxlbmRhcgU5Y3RsMDAkVGV4dENvbnRlbnQkQzAwMSRyZHBIaXNzZVNlbmV0bGVyaVBpeWFzYXNpJGNhbGVuZGFyBTBjdGwwMCRUZXh0Q29udGVudCRDMDAxJGJ0bkhpc3NlU2VuZXRsZXJpUGl5YXNhc2kFL2N0bDAwJFRleHRDb250ZW50JEMwMDEkcmRwQmlzdGVjaFBQR3VubHVrQnVsdGVuBThjdGwwMCRUZXh0Q29udGVudCRDMDAxJHJkcEJpc3RlY2hQUEd1bmx1a0J1bHRlbiRjYWxlbmRhcgU4Y3RsMDAkVGV4dENvbnRlbnQkQzAwMSRyZHBCaXN0ZWNoUFBHdW5sdWtCdWx0ZW4kY2FsZW5kYXIFL2N0bDAwJFRleHRDb250ZW50JEMwMDEkYnRuQmlzdGVjaFBQR3VubHVrQnVsdGVuBTFjdGwwMCRUZXh0Q29udGVudCRDMDAxJHJkcEdlY2ljaUthcGFuaXNCdWx0ZW5sZXJpBTpjdGwwMCRUZXh0Q29udGVudCRDMDAxJHJkcEdlY2ljaUthcGFuaXNCdWx0ZW5sZXJpJGNhbGVuZGFyBTpjdGwwMCRUZXh0Q29udGVudCRDMDAxJHJkcEdlY2ljaUthcGFuaXNCdWx0ZW5sZXJpJGNhbGVuZGFyBTFjdGwwMCRUZXh0Q29udGVudCRDMDAxJGJ0bkdlY2ljaUthcGFuaXNCdWx0ZW5sZXJpBSljdGwwMCRUZXh0Q29udGVudCRDMDAxJHJkcExvdEFsdGlJc2xlbWxlcgUyY3RsMDAkVGV4dENvbnRlbnQkQzAwMSRyZHBMb3RBbHRpSXNsZW1sZXIkY2FsZW5kYXIFMmN0bDAwJFRleHRDb250ZW50JEMwMDEkcmRwTG90QWx0aUlzbGVtbGVyJGNhbGVuZGFyBSljdGwwMCRUZXh0Q29udGVudCRDMDAxJGJ0bkxvdEFsdGlJc2xlbWxlcgUnY3RsMDAkVGV4dENvbnRlbnQkQzAwMSRyZHBNYXJqQmlsZ2lsZXJpBTBjdGwwMCRUZXh0Q29udGVudCRDMDAxJHJkcE1hcmpCaWxnaWxlcmkkY2FsZW5kYXIFMGN0bDAwJFRleHRDb250ZW50JEMwMDEkcmRwTWFyakJpbGdpbGVyaSRjYWxlbmRhcgUnY3RsMDAkVGV4dENvbnRlbnQkQzAwMSRidG5NYXJqQmlsZ2lsZXJpf2ebNDcdsM6PODS8Fa1jsWmEzYQPI30oW3F/8SG4NjI=',
        '__VIEWSTATEGENERATOR': '89B5B9BE',
        '__EVENTVALIDATION': '/wEdABW1a2Q76EXV4+1Ux1gg1H3bLU5LSFAfqzgMkbIe5S8Em8LPcdKM5YQ9NazWRu8BBfqp+k/+qkpn6EWuLu3PnONnkbLkuim/WhEVMtBi8PZScl2GfmI0QsJz0JYHosTK3e6Lj5kBQnrwXfe2uKDUqFEPA796wq25zAqU9izNToACunuwhuT2K4XvNEtbjum10TnRMFzvXQV7rvMCY53Zq2uWRqGPnkT7s61eZPMHG/eACuX9KnPPTLRRLexbzyy2El5M+coJAaYwDOB8JeUAmnwgQSjDOopKHk2ydfqfcJxyJ4UeSZlu9yZhY4isAvp7Df4bB0MnFu7tCjyMCE/FUGS7V2YKjBhigoi8CoiF5h2J02aw/7jGE73hzLdSxCEetKGoxG4COBxk1KotyTjz2D4xqED5tHUhg7rz9Bfsi+TjFiXUBKjG8PAk/CKAxOwqQ76W1EDMtttgo6C/KobAC0ogwV/99IOjbJfM0yLRbAfd0Q==',
        'ctl00$ctl17': '',
        'ctl00$arama$T1DEB820E009$ctl00$ctl00$searchTextBox': '',
        'ctl00$TextContent$C001$rdpHisseSenetleriPiyasasi': datestr1,
        'ctl00$TextContent$C001$rdpHisseSenetleriPiyasasi$dateInput': datestr2,
        'ctl00_TextContent_C001_rdpHisseSenetleriPiyasasi_dateInput_ClientState': '{"enabled":true,"emptyMessage":"","validationText":"'+datestr1+'-00-00-00","valueAsString":"'+datestr1+'-00-00-00","minDateStr":"1980-01-01-00-00-00","maxDateStr":"2099-12-31-00-00-00"}',
        'ctl00_TextContent_C001_rdpHisseSenetleriPiyasasi_calendar_SD': '[[2019,9,9]]',
        'ctl00_TextContent_C001_rdpHisseSenetleriPiyasasi_calendar_AD': '[[1980,1,1],[2099,12,30],[2017,12,1]]',
        'ctl00_TextContent_C001_rdpHisseSenetleriPiyasasi_ClientState': '',
        'ctl00$TextContent$C001$cboHisseSenetleriPiyasasi': '2',
        'ctl00$TextContent$C001$btnHisseSenetleriPiyasasi.x': '10',
        'ctl00$TextContent$C001$btnHisseSenetleriPiyasasi.y': '11',
        'ctl00$TextContent$C001$rdpBistechPPGunlukBulten': '2019-09-05',
        'ctl00$TextContent$C001$rdpBistechPPGunlukBulten$dateInput': '5.9.2019',
        'ctl00_TextContent_C001_rdpBistechPPGunlukBulten_dateInput_ClientState': '{"enabled":true,"emptyMessage":"","validationText":"2019-09-05-00-00-00","valueAsString":"2019-09-05-00-00-00","minDateStr":"2015-11-29-00-00-00","maxDateStr":"2099-12-31-00-00-00"}',
        'ctl00_TextContent_C001_rdpBistechPPGunlukBulten_calendar_SD': '[[2019,9,5]]',
        'ctl00_TextContent_C001_rdpBistechPPGunlukBulten_calendar_AD': '[[2015,11,29],[2099,12,30],[2019,9,14]]',
        'ctl00_TextContent_C001_rdpBistechPPGunlukBulten_ClientState': '{"minDateStr":"2015-11-29-00-00-00","maxDateStr":"2099-12-31-00-00-00"}',
        'ctl00$TextContent$C001$rdpGeciciKapanisBultenleri': '2013-02-14',
        'ctl00$TextContent$C001$rdpGeciciKapanisBultenleri$dateInput': '14.2.2013',
        'ctl00_TextContent_C001_rdpGeciciKapanisBultenleri_dateInput_ClientState': '{"enabled":true,"emptyMessage":"","validationText":"2013-02-14-00-00-00","valueAsString":"2013-02-14-00-00-00","minDateStr":"1980-01-01-00-00-00","maxDateStr":"2013-02-14-00-00-00"}',
        'ctl00_TextContent_C001_rdpGeciciKapanisBultenleri_calendar_SD': '[]',
        'ctl00_TextContent_C001_rdpGeciciKapanisBultenleri_calendar_AD': '[[1980,1,1],[2013,2,14],[2013,2,14]]',
        'ctl00_TextContent_C001_rdpGeciciKapanisBultenleri_ClientState': '{"minDateStr":"1980-01-01-00-00-00","maxDateStr":"2013-02-14-00-00-00"}',
        'ctl00$TextContent$C001$cboGeciciKapanisBultenleri': '2',
        'ctl00$TextContent$C001$rdpLotAltiIslemler': '2015-11-29',
        'ctl00$TextContent$C001$rdpLotAltiIslemler$dateInput': '29.11.2015',
        'ctl00_TextContent_C001_rdpLotAltiIslemler_dateInput_ClientState': '{"enabled":true,"emptyMessage":"","validationText":"2015-11-29-00-00-00","valueAsString":"2015-11-29-00-00-00","minDateStr":"1980-01-01-00-00-00","maxDateStr":"2015-11-29-00-00-00"}',
        'ctl00_TextContent_C001_rdpLotAltiIslemler_calendar_SD': '[]',
        'ctl00_TextContent_C001_rdpLotAltiIslemler_calendar_AD': '[[1980,1,1],[2015,11,29],[2015,11,29]]',
        'ctl00_TextContent_C001_rdpLotAltiIslemler_ClientState': '{"minDateStr":"1980-01-01-00-00-00","maxDateStr":"2015-11-29-00-00-00"}',
        'ctl00$TextContent$C001$rdpMarjBilgileri': '2019-09-14',
        'ctl00$TextContent$C001$rdpMarjBilgileri$dateInput': '14.9.2019',
        'ctl00_TextContent_C001_rdpMarjBilgileri_dateInput_ClientState': '{"enabled":true,"emptyMessage":"","validationText":"2019-09-14-00-00-00","valueAsString":"2019-09-14-00-00-00","minDateStr":"1980-01-01-00-00-00","maxDateStr":"2099-12-31-00-00-00"}',
        'ctl00_TextContent_C001_rdpMarjBilgileri_calendar_SD': '[]',
        'ctl00_TextContent_C001_rdpMarjBilgileri_calendar_AD': '[[1980,1,1],[2099,12,30],[2019,9,14]]',
        'ctl00_TextContent_C001_rdpMarjBilgileri_ClientState': '',
        'ctl00$TextContent$C001$cboMarjBilgileri': '1',
        'ctl00$TextContent$C001$cboMarjBilgileriBistech': '1'
    }
    r = requests.post("https://www.borsaistanbul.com/veriler/verileralt/hisse-senetleri-piyasasi-verileri/bulten-verileri", data=data)
    zipdata = BytesIO()
    try:
        unzip = zipfile.ZipFile(BytesIO(r.content))
    except:
        if "Dosya Bulunamad" in r.text:
            date = date - datedelta
            continue
        else:
            with open("{0}.html".format(date.strftime("%Y%m%d")), "wb") as file:
                file.write(r.content)
            raise
    for filename in unzip.namelist():
        with unzip.open(filename) as unzipped:
            with open("files/{0}{1}".format(filename_, filename[-4:]), "wb") as writefile:
                writefile.write(unzipped.read())
    date = date - datedelta