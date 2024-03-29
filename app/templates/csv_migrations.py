import numpy as np
import pandas as pd
from numpy import genfromtxt
from app import db
from app.models import Users, Movies, Interactions
import os

def Load_Data(file_name):
    data = genfromtxt(file_name, delimiter=',', skip_header=1, converters={0: lambda s: int(s),
                                                                           1: lambda s: int(s),
                                                                           2: lambda s: int(s),
                                                                           4: lambda s: int(s)
                                                                           })
    return data.tolist()

def Load_Movies(file_name):
    # data = genfromtxt(file_name, delimiter=',', skip_header=1, converters={0: lambda s: int(s)})
    # return data.tolist()
    data = pd.read_csv(os.path.join(os.path.dirname(__file__),'edited_ratings_subset_larger.csv'))
    data = pd.DataFrame(data)
    data = data[['movieId','movie_idx']]
    df_rm = pd.DataFrame.drop_duplicates(data)
    movies = pd.read_csv(file_name)
    movies = pd.DataFrame(movies)

    merge = pd.merge(df_rm, movies, on='movieId', how='left')
    movie_db = merge[['movie_idx','title','genres']]
    movie_db = movie_db.sort_values(by = 'movie_idx', ascending = True)
    #movie_db = movie_db[:10]
    movie_db['thumbnail'] = [
    "Movies/forestgump.jpeg",
    "Movies/shawshankRedemption.jpeg",
    "Movies/pulpFiction.jpg",
    "Movies/sotl.jpg",
    "Movies/matrix.jpeg",
    "Movies/starWars.jpg",
    "Movies/jurassicPark.jpg",
    "Movies/braveheart.jpg",
    "Movies/terminator2.jpg",
    "Movies/schlinderList.jpg",
    "Movies/fightclub.jpg",
    "Movies/toystory.jpg",
    "Movies/SW5.jpg",
    "Movies/tus.jpg",
    "Movies/american_beauty.jpg",
    "Movies/7.jpg",
    "Movies/id.jpg",
    "Movies/apollo13.jpg",
    "Movies/rotla.jpg",
    "Movies/lor.jpg",
    "Movies/sv6.jpeg",
    "Movies/godfather.jpg",
    "Movies/tf.jpg",
    "Movies/batman.jpg",
    "Movies/spr.jpg",
    "Movies/lor2.jpg",
    "Movies/lor3.jpg",
    "Movies/aladdin.jpg",
    "Movies/fargo.jpg",
    "Movies/tss.jpg",
    "Movies/tl.jpg",
    "Movies/12_monkeys.jpg",
    "Movies/lk.jpeg",
    "Movies/btf.jpg",
    "Movies/speed.jpeg",
    "Movies/gladiator.jpg",
    "Movies/Shrek.jpg",
    "Movies/mib.jpg",
    "Movies/dww.jpg",
    "Movies/mi.jpg",
    "Movies/avpd.jpg",
    "Movies/memento.jpg",
    "Movies/mask.jpg",
    "Movies/tdk.jpg",
    "Movies/potc.jpg",
    "Movies/alien.jpg",
    "Movies/bab.jpg",
    "Movies/die_hard.jpeg",
    "Movies/md.jpg",
    "Movies/dhv.jpg",
    "Movies/ghd.jpg",
    "Movies/Inception.jpg",
    "Movies/tpb.jpg",
    "Movies/gwh.jpg",
    "Movies/fnd_nemo.jpg",
    "Movies/stargate.jpg",
    "Movies/indiana_jones.jpg",
    "Movies/star_wars1.jpg",
    "Movies/titanic.jpg",
    "Movies/batman_forever.jpg",
    "Movies/monty_python.jpg",
    "Movies/pretty_woman.jpg",
    "Movies/dnd.jpg",
    "Movies/xmen.jpg",
    "Movies/leon.jpg",
    "Movies/ofocn.jpg",
    "Movies/golden_eye.jpg",
    "Movies/monsterinc.jpg",
    "Movies/res_dogs.jpg",
    "Movies/terminator.jpg",
    "Movies/killbill1.jpg",
    "Movies/esosm.jpg",
    "Movies/ahx.jpg",
    "Movies/godfather2.jpg",
    "Movies/babe.jpg",
    "Movies/goodfellas.jpg",
    "Movies/aliens.jpg",
    "Movies/truman_show.jpg",
    "Movies/incredibles.jpg",
    "Movies/blade_runner.jpg",
    "Movies/twister.jpg",
    "Movies/beautiful_mind.jpg",
    "Movies/et.jpg",
    "Movies/spider_man.jpg",
    "Movies/therock.jpeg",
    "Movies/austin_powers.jpg",
    "Movies/clockwork_orange.jpg",
    "Movies/ghost_busters.jpg",
    "Movies/minority_report.jpg",
    "Movies/amelie.jpg",
    "Movies/wwcf.jpg",
    "Movies/oceans11.jpg",
    "Movies/batman_begins.jpg",
    "Movies/home_alone.jpg",
    "Movies/tfe.jpg",
    "Movies/waterworld.jpg",
    "Movies/ghost.jpg",
    "Movies/cmiyc.jpeg",
    "Movies/tbc.jpg",
    "Movies/the_net.jpg",
    "Movies/tbi.jpg",
    "Movies/tgm.jpg",
    "Movies/cpd.jpg",
    "Movies/cthd.jpg",
    "Movies/jumanji.jpg",
    "Movies/kb2.jpg",
    "Movies/shining.jpg",
    "Movies/iwv.jpg",
    "Movies/2001.jpg",
    "Movies/donnie_darko.jpg",
    "Movies/ferris_buellers.jpg",
    "Movies/ijtd.jpg",
    "Movies/star_trek_gen.jpg",
    "Movies/apoc_now.jpg",
    "Movies/departed.jpg",
    "Movies/hp_st.jpg",
    "Movies/tbl.jpg",
    "Movies/sleepless_seattle.jpg",
    "Movies/mary.jpg",
    "Movies/up.jpg",
    "Movies/clerks.jpg",
    "Movies/clueless.jpg",
    "Movies/wall_e.jpg",
    "Movies/taxi_driver.jpg",
    "Movies/four_weddings.jpg",
    "Movies/crimson_tide.jpg",
    "Movies/american_pie.jpg",
    "Movies/heat.jpg",
    "Movies/born_to_kill.jpg",
    "Movies/hp_cs.jpg",
    "Movies/trainspotting.jpg",
    "Movies/outbreak.jpg",
    "Movies/cliffhanger.jpg",
    "Movies/the_firm.jpg",
    "Movies/austin_powers.jpg",
    "Movies/casablanca.jpg",
    "Movies/cast_away.jpg",
    "Movies/v_for_v.jpg",
    "Movies/being_john_malkovich.jpg",
    "Movies/happy_gilmore.jpg",
    "Movies/while_you.jpg",
    "Movies/lac.jpg",
    "Movies/roger_rabbit.jpg",
    "Movies/dsl.jpg",
    "Movies/toy_story2.jpg",
    "Movies/avatar.jpg",
    "Movies/requim.jpg",
    "Movies/as_good.jpg",
    "Movies/matrix_rel.jpg",
    "Movies/rain_man.jpg",
    "Movies/office_space.jpg",
    "Movies/o_brother.jpg",
    "Movies/iron_man.jpg",
    "Movies/hp_poa.jpeg",
    "Movies/snatch.jpg",
    "Movies/nightmare_bf.jpg",
    "Movies/wizard_oz.jpg",
    "Movies/nbk.jpg",
    "Movies/armageddon.jpg",
    "Movies/sw2.jpg",
    "Movies/shrek2.jpg",
    "Movies/bugs_life.jpg",
    "Movies/sil.jpg",
    "Movies/beetle_juice.jpg",
    "Movies/big.jpg",
    "Movies/stand_by_me.jpg",
    "Movies/stfc.jpg",
    "Movies/mtp.jpg",
    "Movies/jaws.jpg",
    "Movies/hunt.jpg",
    "Movies/the_prestige.jpg",
    "Movies/monty.jpg",
    "Movies/get_shorty.jpg",
    "Movies/back2_future2.jpeg",
    "Movies/total_recall.jpg",
    "Movies/inglo_bes.jpg",
    "Movies/life.jpg",
    "Movies/ace_ventura.jpg",
    "Movies/the_mummy.jpg",
    "Movies/airplane.jpg",
    "Movies/spirited_away.jpg",
    "Movies/b2f2.jpg",
    "Movies/mars_attacks.jpg",
    "Movies/birdcage.jpg",
    "Movies/dps.jpg",
    "Movies/gattaca.jpg",
    "Movies/ice_age.jpg",
    "Movies/when_harry.jpg",
    "Movies/jerry.jpg",
    "Movies/blues_brother.jpg",
    "Movies/rear_window.jpeg",
    "Movies/addams_family.jpg",
    "Movies/broken_arrow.jpg",
    "Movies/sin_city.jpg",
    "Movies/psycho.jpg",
    "Movies/almost_famous.jpg",
    "Movies/top_gun.jpg",
    "Movies/casino.jpeg",
    "Movies/contact.jpg",
    "Movies/nutty_prof.jpg"
]
    
    movie_db['watchlink'] = [
    "https://www.youtube.com/watch?v=uPIEn0M8su0",
    "https://www.youtube.com/watch?v=6hB3S9bIaco",
    "https://www.youtube.com/watch?v=s7EdQ4FqbhY",
    "https://www.youtube.com/watch?v=RuX2MQeb8UM&t=2s",
    "https://www.youtube.com/watch?v=vKQi3bBA1y8",
    "https://www.youtube.com/watch?v=367FSjWvNB4",
    "https://www.youtube.com/watch?v=lc0UehYemQA",
    "https://www.youtube.com/watch?v=1NJO0jxBtMo",
    "https://www.youtube.com/watch?v=-W8CegO_Ixw",
    "https://www.youtube.com/watch?v=gG22XNhtnoY",
    "https://www.youtube.com/watch?v=qtRKdVHc-cE",
    "https://www.youtube.com/watch?v=v-PjgYDrg70",
    "https://www.youtube.com/watch?v=JNwNXF9Y6kY",
    "https://www.youtube.com/watch?v=oiXdPolca5w",
    "https://www.youtube.com/watch?v=y4DvNwt0FSM",
    "https://www.youtube.com/watch?v=znmZoVkCjpI",
    "https://www.youtube.com/watch?v=B1E7h3SeMDk",
    "https://www.youtube.com/watch?v=KtEIMC58sZo",
    "https://www.youtube.com/watch?v=Rh_BJXG1-44",
    "https://www.youtube.com/watch?v=V75dMMIW2B4",
    "https://www.youtube.com/watch?v=5UfA_aKBGMc",
    "https://www.youtube.com/watch?v=sY1S34973zA&t=2s",
    "https://www.youtube.com/watch?v=ETPVU0acnrE",
    "https://www.youtube.com/watch?v=dgC9Q0uhX70",
    "https://www.youtube.com/watch?v=zwhP5b4tD6g",
    "https://www.youtube.com/watch?v=LbfMDwc4azU",
    "https://www.youtube.com/watch?v=r5X-hFf6Bwo",
    "https://www.youtube.com/watch?v=eTjHiQKJUDY",
    "https://www.youtube.com/watch?v=h2tY82z3xXU",
    "https://www.youtube.com/watch?v=3-ZP95NF_Wk",
    "https://www.youtube.com/watch?v=3B7HG8_xbDw",
    "https://www.youtube.com/watch?v=15s4Y9ffW_o",
    "https://www.youtube.com/watch?v=hY7xBISLBIA",
    "https://www.youtube.com/watch?v=qvsgGtivCgs",
    "https://www.youtube.com/watch?v=8piqd2BWeGI",
    "https://www.youtube.com/watch?v=uvbavW31adA",
    "https://www.youtube.com/watch?v=CwXOrWvPBPk",
    "https://www.youtube.com/watch?v=1Q4mhYF9aQQ",
    "https://www.youtube.com/watch?v=uc8NMbrW7mI",
    "https://www.youtube.com/watch?v=Ohws8y572KE",
    "https://www.youtube.com/watch?v=ZGlFA-miRLw",
    "https://www.youtube.com/watch?v=HDWylEQSwFo",
    "https://www.youtube.com/watch?v=LZl69yk5lEY",
    "https://www.youtube.com/watch?v=EXeTwQWrcwY",
    "https://www.youtube.com/watch?v=naQr0uTrH_s",
    "https://www.youtube.com/watch?v=LjLamj-b0I8",
    "https://www.youtube.com/watch?v=iurbZwxKFUE",
    "https://www.youtube.com/watch?v=2TQ-pOvI6Xo",
    "https://www.youtube.com/watch?v=3euGQ7-brs4",
    "https://www.youtube.com/watch?v=gQ0uSh2Hgcs",
    "https://www.youtube.com/watch?v=GncQtURdcE4",
    "https://www.youtube.com/watch?v=YoHD9XEInc0&t=6s",
    "https://www.youtube.com/watch?v=O3CIXEAjcc8",
    "https://www.youtube.com/watch?v=PaZVjZEFkRs",
    "https://www.youtube.com/watch?v=9oQ628Seb9w",
    "https://www.youtube.com/watch?v=kiJtZUPvJxY",
    "https://www.youtube.com/watch?v=sagmdpkWUqc&t=1s",
    "https://www.youtube.com/watch?v=uMoSnrd7i5c",
    "https://www.youtube.com/watch?v=ezcvpLIyifU",
    "https://www.youtube.com/watch?v=ROLvjRB4E_Q",
    "https://www.youtube.com/watch?v=urRkGvhXc8w",
    "https://www.youtube.com/watch?v=2EBAVoN8L_U",
    "https://www.youtube.com/watch?v=l13yPhimE3o",
    "https://www.youtube.com/watch?v=nbNcULQFojc",
    "https://www.youtube.com/watch?v=aNQqoExfQsg",
    "https://www.youtube.com/watch?v=OXrcDonY-B8&t=2s",
    "https://www.youtube.com/watch?v=8Zw8ylP4buA",
    "https://www.youtube.com/watch?v=cvOQeozL4S0",
    "https://www.youtube.com/watch?v=vayksn4Y93A",
    "https://www.youtube.com/watch?v=k64P4l2Wmeg",
    "https://www.youtube.com/watch?v=7kSuas6mRpk",
    "https://www.youtube.com/watch?v=07-QBnEkgXU",
    "https://www.youtube.com/watch?v=XfQYHqsiN5g",
    "https://www.youtube.com/watch?v=9O1Iy9od7-A",
    "https://www.youtube.com/watch?v=yuzXPzgBDvo",
    "https://www.youtube.com/watch?v=2ilzidi_J8Q",
    "https://www.youtube.com/watch?v=XKSQmYUaIyE",
    "https://www.youtube.com/watch?v=loTIzXAS7v4",
    "https://www.youtube.com/watch?v=-UaGUdNJdRQ&t=2s",
    "https://www.youtube.com/watch?v=eogpIG53Cis",
    "https://www.youtube.com/watch?v=OgG2jfBfLzI&t=1s",
    "https://www.youtube.com/watch?v=9wZM7CQY130",
    "https://www.youtube.com/watch?v=qYAETtIIClk",
    "https://www.youtube.com/watch?v=TYMMOjBUPMM",
    "https://www.youtube.com/watch?v=jGVJx5mOtL8",
    "https://www.youtube.com/watch?v=mYVb4OLk4NQ",
    "https://www.youtube.com/watch?v=SPRzm8ibDQ8",
    "https://www.youtube.com/watch?v=6hDkhw5Wkas",
    "https://www.youtube.com/watch?v=lG7DGMgfOb8",
    "https://www.youtube.com/watch?v=HUECWi5pX7o",
    "https://www.youtube.com/watch?v=2cBja3AbahY",
    "https://www.youtube.com/watch?v=imm6OR605UI",
    "https://www.youtube.com/watch?v=neY2xVmOfUM",
    "https://www.youtube.com/watch?v=jEDaVHmw7r4",
    "https://www.youtube.com/watch?v=fQ9RqgcR24g",
    "https://www.youtube.com/watch?v=NpKbULrB9Z8",
    "https://www.youtube.com/watch?v=SMlZDJ6-G3U",
    "https://www.youtube.com/watch?v=xas1UyTiVUw",
    "https://www.youtube.com/watch?v=BSXBvor47Zs",
    "https://www.youtube.com/watch?v=46qKHq7REI4",
    "https://www.youtube.com/watch?v=FpKaB5dvQ4g",
    "https://www.youtube.com/watch?v=Ki4haFrqSrw",
    "https://www.youtube.com/watch?v=3sG1tGbpT7c",
    "https://www.youtube.com/watch?v=gLpZ_5bHmo8",
    "https://www.youtube.com/watch?v=DvQ-PGUr6SM",
    "https://www.youtube.com/watch?v=WTt8cCIvGYI",
    "https://www.youtube.com/watch?v=5Cb3ik6zP2I",
    "https://www.youtube.com/watch?v=sCmYN6TLd8A",
    "https://www.youtube.com/watch?v=Z2UWOeBcsJI",
    "https://www.youtube.com/watch?v=ZZyBaFYFySk",
    "https://www.youtube.com/watch?v=K-X2XzKqBiE",
    "https://www.youtube.com/watch?v=98Nkqlp_hrg",
    "https://www.youtube.com/watch?v=FDXgeo_1w8k",
    "https://www.youtube.com/watch?v=FTjG-Aux_yQ",
    "https://www.youtube.com/watch?v=iQpb1LoeVUc",
    "https://www.youtube.com/watch?v=VyHV0BRtdxo",
    "https://www.youtube.com/watch?v=cd-go0oBF4Y",
    "https://www.youtube.com/watch?v=-Lj2U-cmyek",
    "https://www.youtube.com/watch?v=eGjXwDYpOLE",
    "https://www.youtube.com/watch?v=ORFWdXl_zJ4",
    "https://www.youtube.com/watch?v=Mlfn5n-E2WE",
    "https://www.youtube.com/watch?v=RS0KyTZ3Ie4",
    "https://www.youtube.com/watch?v=qGBZWbg_26A",
    "https://www.youtube.com/watch?v=UUxD4-dEzn0",
    "https://www.youtube.com/watch?v=g-HeV8Z6iXc",
    "https://www.youtube.com/watch?v=iS4I2Z1RBIw",
    "https://www.youtube.com/watch?v=iUZ3Yxok6N8",
    "https://www.youtube.com/watch?v=2GfZl4kuVNI",
    "https://www.youtube.com/watch?v=Ks_MbPPkhmA",
    "https://www.youtube.com/watch?v=1bq0qff4iF8",
    "https://www.youtube.com/watch?v=8LuxOYIpu-I",
    "https://www.youtube.com/watch?v=Y5povsMKfT4",
    "https://www.youtube.com/watch?v=OZALfW1ohRI",
    "https://www.youtube.com/watch?v=Auxb3l4Y8j0",
    "https://www.youtube.com/watch?v=5vsANcS4Ml8",
    "https://www.youtube.com/watch?v=BkL9l7qovsE",
    "https://www.youtube.com/watch?v=qGuOZPwLayY",
    "https://www.youtube.com/watch?v=lSA7mAHolAw",
    "https://www.youtube.com/watch?v=2UuRFr0GnHM",
    "https://www.youtube.com/watch?v=y1emDAYCfVQ",
    "https://www.youtube.com/watch?v=nsJxyUvkB_E",
    "https://www.youtube.com/watch?v=6sOXrY5yV4g",
    "https://www.youtube.com/watch?v=gpDaNqSXxp0",
    "https://www.youtube.com/watch?v=IE9CmX15PYA",
    "https://www.youtube.com/watch?v=xNWSGRD5CzU",
    "https://www.youtube.com/watch?v=6ziBFh3V1aM",
    "https://www.youtube.com/watch?v=QBwzN4v1vA0",
    "https://www.youtube.com/watch?v=rrRl2QQKkI8",
    "https://www.youtube.com/watch?v=kYzz0FSgpSU",
    "https://www.youtube.com/watch?v=mlNwXuHUA8I",
    "https://www.youtube.com/watch?v=dMIrlP61Z9s",
    "https://www.youtube.com/watch?v=n9UlbxlM5nE",
    "https://www.youtube.com/watch?v=8ugaeA-nMTc",
    "https://www.youtube.com/watch?v=lAxgztbYDbs",
    "https://www.youtube.com/watch?v=ni4tEtuTccc",
    "https://www.youtube.com/watch?v=wr6N_hZyBCk",
    "https://www.youtube.com/watch?v=njdreZRjvpc",
    "https://www.youtube.com/watch?v=XpLKNclOtLg",
    "https://www.youtube.com/watch?v=8-8eEniEfgU",
    "https://www.youtube.com/watch?v=gYbW1F_c9eM",
    "https://www.youtube.com/watch?v=V6X5ti4YlG8",
    "https://www.youtube.com/watch?v=Ljk2YJ53_WI",
    "https://www.youtube.com/watch?v=gk1rTKB6ZF8",
    "https://www.youtube.com/watch?v=ickbVzajrk0",
    "https://www.youtube.com/watch?v=BGDTNhHYJ94",
    "https://www.youtube.com/watch?v=oYTfYsODWQo",
    "https://www.youtube.com/watch?v=LlBdRZiTUjk",
    "https://www.youtube.com/watch?v=CXqxP-bUC7I",
    "https://www.youtube.com/watch?v=U1fu_sA7XhE",
    "https://www.youtube.com/watch?v=3C2tE7vjdHk",
    "https://www.youtube.com/watch?v=ijXruSzfGEc",
    "https://www.youtube.com/watch?v=IEh1HFFfuMo",
    "https://www.youtube.com/watch?v=fvBx0x9IJnk",
    "https://www.youtube.com/watch?v=EYkguxpqsrg",
    "https://www.youtube.com/watch?v=WFMLGEHdIjE",
    "https://www.youtube.com/watch?v=KnrRy6kSFF0",
    "https://www.youtube.com/watch?v=8CTjcVr9Iao",
    "https://www.youtube.com/watch?v=T8aos7_L4kA",
    "https://www.youtube.com/watch?v=h3ptPtxWJRs",
    "https://www.youtube.com/watch?v=jHjPY1jG7lo",
    "https://www.youtube.com/watch?v=ByXuk9QqQkk",
    "https://www.youtube.com/watch?v=MdENmefJRpw",
    "https://www.youtube.com/watch?v=DqtjHWlM4lQ",
    "https://www.youtube.com/watch?v=P7FcPlt8hHc",
    "https://www.youtube.com/watch?v=4lj185DaZ_o",
    "https://www.youtube.com/watch?v=DO_x-po_Nsc",
    "https://www.youtube.com/watch?v=cMfeWyVBidk",
    "https://www.youtube.com/watch?v=vmSpCLefjnw",
    "https://www.youtube.com/watch?v=rCCaTPY-z4Q",
    "https://www.youtube.com/watch?v=2HCR4c1zPyk",
    "https://www.youtube.com/watch?v=m01YktiEZCw",
    "https://www.youtube.com/watch?v=EisokUNMfeA",
    "https://www.youtube.com/watch?v=De29KipeZQw",
    "https://www.youtube.com/watch?v=T2Dj6ktPU5c",
    "https://www.youtube.com/watch?v=Wz719b9QUqY",
    "https://www.youtube.com/watch?v=aQXh_AaJXaM",
    "https://www.youtube.com/watch?v=xa_z57UatDY",
    "https://www.youtube.com/watch?v=EJXDMwGWhoA",
    "https://www.youtube.com/watch?v=_1spaRVuhVQ",
    "https://www.youtube.com/watch?v=o3wJ-jzZqBw"
    ]

    users = Interactions.query.with_entities(Interactions.userid).all()
    users = [int(u) for u, in users]
    movies = Interactions.query.with_entities(Interactions.movieid).all()
    movies = [int(m) for m, in movies]
    ratings = Interactions.query.with_entities(Interactions.rating).all()
    ratings = [r for r, in ratings]
    df = {'userid':users,'movieid':movies,'rating':ratings}
    df = pd.DataFrame(df)
    means = df.groupby(['movieid'],as_index = False).mean().sort_values(by = 'movieid')
    mean_ratings = [x for x in list(means['rating'].values)]

    movie_db['avg_rating'] = mean_ratings

    n_ratings = list(df.groupby(['movieid'],as_index = False).count()['userid'].values)
    n_ratings = [int(n) for n in n_ratings]
    movie_db['N_ratings'] = n_ratings

    return movie_db


# file_name = '/edited_ratings_subset_larger.csv'
# data = Load_Data(file_name)
def migrate_interactions():
    data = Load_Data(os.path.join(os.path.dirname(__file__),'edited_ratings_subset_larger.csv'))
    existing_records = Interactions.query.all()
    if existing_records is not None:
        for i in existing_records:
            db.session.delete(i)
        db.session.commit()
    for i in data:
        record = Interactions(
            userid = i[1],
            movieid = i[4],
            rating = i[3]
        )
        db.session.add(record)
    db.session.commit()
    return "Interactions Migration complete"

def migrate_customers():
    data = [
    {'userid' : 0, 'username' : 'Jeremy', 'email' : 'jeremy@gmail.com', 'password' : 'pass@12'},
    {'userid' : 1, 'username' : 'Mike', 'email' : 'mike@gmail.com', 'password' : 'pass@13'},
    {'userid' : 2, 'username' : 'Jason', 'email' : 'jason@gmail.com', 'password' : 'pass@14'},
    {'userid' : 3, 'username' : 'Mark', 'email' : 'mark@gmail.com', 'password' : 'pass@15'},
    {'userid' : 4, 'username' : 'Asha', 'email' : 'asha@gmail.com', 'password' : 'pass@16'},
    {'userid' : 5, 'username' : 'Maria', 'email' : 'maria@gmail.com', 'password' : 'pass@17'},
    {'userid' : 6, 'username' : 'Jesus', 'email' : 'jesus@gmail.com', 'password' : 'pass@18'},
    {'userid' : 7, 'username' : 'Murphy', 'email' : 'murphy@gmail.com', 'password' : 'pass@19'},
    {'userid' : 8, 'username' : 'Alia', 'email' : 'alia@gmail.com', 'password' : 'pass@20'},
    {'userid' : 9, 'username' : 'Malik', 'email' : 'malik@gmail.com', 'password' : 'pass@21'},
    {'userid' : 10, 'username' : 'Jannat', 'email' : 'jannat@gmail.com', 'password' : 'pass@22'},
    {'userid' : 11, 'username' : 'Mylo', 'email' : 'mylo@gmail.com', 'password' : 'pass@23'},
    {'userid' : 12, 'username' : 'Ashley', 'email' : 'ashley@gmail.com', 'password' : 'pass@24'},
    {'userid' : 13, 'username' : 'Masan', 'email' : 'masan@gmail.com', 'password' : 'pass@25'},
    {'userid' : 14, 'username' : 'Arshad', 'email' : 'arshad@gmail.com', 'password' : 'pass@26'},
    {'userid' : 15, 'username' : 'Phil', 'email' : 'phil@gmail.com', 'password' : 'pass@27'},
    {'userid' : 16, 'username' : 'Jack', 'email' : 'jack@gmail.com', 'password' : 'pass@28'},
    {'userid' : 17, 'username' : 'Michelle', 'email' : 'michelle@gmail.com', 'password' : 'pass@29'},
    {'userid' : 18, 'username' : 'Jared', 'email' : 'jared@gmail.com', 'password' : 'pass@30'},
    {'userid' : 19, 'username' : 'Maitree', 'email' : 'maitree@gmail.com', 'password' : 'pass@31'},
    {'userid' : 20, 'username' : 'Filza', 'email' : 'filza@gmail.com', 'password' : 'pass@32'},
    {'userid' : 21, 'username' : 'Rukh', 'email' : 'rukh@gmail.com', 'password' : 'pass@33'},
    {'userid' : 22, 'username' : 'Rose', 'email' : 'rose@gmail.com', 'password' : 'pass@34'},
    {'userid' : 23, 'username' : 'Black', 'email' : 'black@gmail.com', 'password' : 'pass@35'},
    {'userid' : 24, 'username' : 'White', 'email' : 'white@gmail.com', 'password' : 'pass@36'},
    {'userid' : 25, 'username' : 'Purple', 'email' : 'purple@gmail.com', 'password' : 'pass@37'},
    {'userid' : 26, 'username' : 'Shaun', 'email' : 'shaun@gmail.com', 'password' : 'pass@38'},
    {'userid' : 27, 'username' : 'Joe', 'email' : 'joe@gmail.com', 'password' : 'pass@39'},
    {'userid' : 28, 'username' : 'Varun', 'email' : 'varun@gmail.com', 'password' : 'pass@40'},
    {'userid' : 29, 'username' : 'Fleming', 'email' : 'fleming@gmail.com', 'password' : 'pass@41'},
    {'userid' : 30, 'username' : 'Ananya', 'email' : 'ananya@gmail.com', 'password' : 'pass@42'},
    {'userid' : 31, 'username' : 'Wang', 'email' : 'wang@gmail.com', 'password' : 'pass@43'},
    {'userid' : 32, 'username' : 'Andrew', 'email' : 'andrew@gmail.com', 'password' : 'pass@44'},
    {'userid' : 33, 'username' : 'Jazzy', 'email' : 'jazzy@gmail.com', 'password' : 'pass@45'},
    {'userid' : 34, 'username' : 'Antara', 'email' : 'antara@gmail.com', 'password' : 'pass@46'},
    {'userid' : 35, 'username' : 'Sitara', 'email' : 'sitara@gmail.com', 'password' : 'pass@47'},
    {'userid' : 36, 'username' : 'Lockie', 'email' : 'lockie@gmail.com', 'password' : 'pass@48'},
    {'userid' : 37, 'username' : 'Virat', 'email' : 'virat@gmail.com', 'password' : 'pass@49'},
    {'userid' : 38, 'username' : 'Anjum', 'email' : 'anjum@gmail.com', 'password' : 'pass@50'},
    {'userid' : 39, 'username' : 'Lokesh', 'email' : 'lokesh@gmail.com', 'password' : 'pass@51'},
    {'userid' : 40, 'username' : 'Sarah', 'email' : 'sarah@gmail.com', 'password' : 'pass@52'},
    {'userid' : 41, 'username' : 'Jerome', 'email' : 'jerome@gmail.com', 'password' : 'pass@53'},
    {'userid' : 42, 'username' : 'Steven', 'email' : 'steven@gmail.com', 'password' : 'pass@54'},
    {'userid' : 43, 'username' : 'David', 'email' : 'david@gmail.com', 'password' : 'pass@55'},
    {'userid' : 44, 'username' : 'Anderson', 'email' : 'anderson@gmail.com', 'password' : 'pass@56'},
    {'userid' : 45, 'username' : 'Stuart', 'email' : 'stuart@gmail.com', 'password' : 'pass@57'},
    {'userid' : 46, 'username' : 'Shikhar', 'email' : 'shikhar@gmail.com', 'password' : 'pass@58'},
    {'userid' : 47, 'username' : 'Rohit', 'email' : 'rohit@gmail.com', 'password' : 'pass@59'},
    {'userid' : 48, 'username' : 'Priya', 'email' : 'priya@gmail.com', 'password' : 'pass@60'},
    {'userid' : 49, 'username' : 'Dhoni', 'email' : 'dhoni@gmail.com', 'password' : 'pass@61'}
    ]
    existing_records = Users.query.all()
    if existing_records is not None:
        for u in existing_records:
            db.session.delete(u)
        db.session.commit()
    for i in data:
        record = Users(
            userid = i['userid'],
            username = i['username'],
            email = i['email'],
            password = i['password']
        )
        db.session.add(record)
    db.session.commit()
    return "Users Migration complete"

def migrate_movies():
    data = Load_Movies(os.path.join(os.path.dirname(__file__),'movies.csv'))
    existing_records = Movies.query.all()
    if existing_records is not None:
        for m in existing_records:
            db.session.delete(m)
        db.session.commit()

    for i in range(200):
        record = Movies(
            movieid = int(data.iloc[i,:][0]),
            moviename = data.iloc[i,:][1],
            genre = data.iloc[i,:][2],
            thumbnail = data.iloc[i,:][3],
            watchlink = data.iloc[i,:][4],
            avg_rating = data.iloc[i,:][5],
            N_ratings = int(data.iloc[i,:][6])
        )
        db.session.add(record)
    db.session.commit()
    return "Movies Migration complete"
