#
# This data is from Fujiwara et al.'s work in VIS 2021
#

#
# Takanori Fujiwara generated this file from the information available in
# https://www.ppic.org/data-set/ppic-statewide-survey-data-2018/
#
import numpy as np

metainfo = {
    'id': {
        'full': 'ID',
        'short': 'id',
        'use': 0,
        'code_num': lambda a: int(a),
        'type': 'categorical'
    },
    'version': {
        'full': 'Interview Version',
        'short': 'version',
        'use': 0,
        'code_text': {
            1: 'Landline',
            2: 'Cellphone'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 0,
            2: 1
        }[a],
        'type': 'binary'
    },
    'county': {
        'full': 'S2c. In which California county do you live?',
        'short': 'county',
        'use': 0,
        'code_text': {
            1: 'Alameda',
            2: 'Alpine',
            3: 'Amador',
            4: 'Butte',
            5: 'Calaveras',
            6: 'Colusa',
            7: 'Contra Costa',
            8: 'Del Norte',
            9: 'El Dorado',
            10: 'Fresno',
            11: 'Glenn',
            12: 'Humboldt',
            13: 'Imperial',
            14: 'Inyo',
            15: 'Kern',
            16: 'Kings',
            17: 'Lake',
            18: 'Lassen',
            19: 'Los Angeles',
            20: 'Madera',
            21: 'Marin',
            22: 'Mariposa',
            23: 'Mendocino',
            24: 'Merced',
            25: 'Modoc',
            26: 'Mono',
            27: 'Monterey',
            28: 'Napa',
            29: 'Nevada',
            30: 'Orange',
            31: 'Placer',
            32: 'Plumas',
            33: 'Riverside',
            34: 'Sacramento',
            35: 'San Benito',
            36: 'San Bernardino',
            37: 'San Diego',
            38: 'San Francisco',
            39: 'San Joaquin',
            40: 'San Luis Obispo',
            41: 'San Mateo',
            42: 'Santa Barbara',
            43: 'Santa Clara',
            44: 'Santa Cruz',
            45: 'Shasta',
            46: 'Sierra',
            47: 'Siskiyou',
            48: 'Solano',
            49: 'Sonoma',
            50: 'Stanislaus',
            51: 'Sutter',
            52: 'Tehama',
            53: 'Trinity',
            54: 'Tulare',
            55: 'Tuolumne',
            56: 'Ventura',
            57: 'Yolo',
            58: 'Yuba'
        },
        'code_num': lambda a: int(a),
        'type': 'categorical'
    },
    'q1': {
        'full':
        'Q1. First, overall, do you approve or disapprove of the way that Jerry Brown is handling his job as governor of California?',
        'short': 'approve_brown',
        'use': 1,
        'code_text': {
            1: 'approve',
            2: 'disapprove',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q2': {
        'full':
        'Q2. Overall, do you approve or disapprove of the way that the California Legislature is handling its job?',
        'short': 'approve_legislature',
        'use': 1,
        'code_text': {
            1: 'approve',
            2: 'disapprove',
            8: '(vol) don’t know',
            9: '(vol) refuse '
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q3': {
        'full':
        'Q3. Turning to economic conditions in California, do you think that during the next 12 months we will have good times financially or bad times?',
        'short': 'opinion_cal_econ',
        'use': 1,
        'code_text': {
            1: 'good times',
            2: 'bad times',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q4': {
        'full':
        'Q4. Next, some people are registered to vote and others are not. Are you ABSOLUTELY CERTAIN that you are registered to vote in California?',
        'short': 'vote_register',
        'use': 0,
        'code_text': {
            1: 'yes [ASK Q4a]',
            2: 'no [SKIP TO party]',
            8: 'don’t know (volunteered) [SKIP TO party]',
            9: 'refuse (volunteered) [SKIP TO party]'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q4a': {
        'full': 'Party',
        'short': 'party',
        'use': 1,
        'code_text': {
            1: 'Democrat',
            2: 'Republican',
            3: 'another party (specify)',
            4: 'decline-to-state or independent',
            5: 'registered, don’t know party',
            6: 'not registered',
            8: 'don’t know (volunteered)',
            9: 'refuse (volunteered) '
        },
        'code_num': lambda a: 999 if np.isnan(a) or int(a) == 9 else int(a),
        'type': 'categorical'
    },
    'q5': {
        'full':
        'Q5. Would you call yourself a strong Democrat or not a very strong Democrat?',
        'short': 'strong_democrat',
        'use': 0,
        'code_text': {
            1: 'strong',
            2: 'not very strong',
            8: 'don’t know (volunteered)',
            9: 'refuse (volunteered)'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q5a': {
        'full':
        'Q5a. Would you call yourself a strong Republican or not a very strong Republican?',
        'short': 'strong_republican',
        'use': 0,
        'code_text': {
            1: 'strong',
            2: 'not very strong',
            8: 'don’t know (volunteered)',
            9: 'refuse (volunteered)'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q5b': {
        'full':
        'Q5b. Do you think of yourself as closer to the Republican Party or Democratic Party?',
        'short':
        'close_polotical_party',
        'use':
        0,
        'code_text': {
            1: 'Republican Party',
            2: 'Democratic Party',
            3: 'neither (volunteered)',
            8: 'don’t know (volunteered)',
            9: 'refuse (volunteered)'
        },
        'code_num':
        lambda a: 999 if np.isnan(a) or int(a) == 8 or int(a) == 9 else int(a),
        'type':
        'categorical'
    },
    'q6': {
        'full':
        'Q6. If the November 6th election for governor were being held today, would you vote for [ROTATE] [1] John Cox, a Republican, [OR] [2] Gavin Newsom, a Democrat?',
        'short': 'governor_choice',
        'use': 0,  ##
        'code_text': {
            1: 'John Cox, a Republican',
            2: 'Gavin Newsom, a Democrat',
            3: '(vol) neither/would not vote for governor',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 0,
            2: 1,
            3: 999,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q7': {
        'full':
        "Q7. How closely are you following news about candidates for the 2018 governor's election- very closely, fairly closely, not too closely, or not at all closely?",
        'short': 'following_governor_election_news',
        'use': 1,
        'code_text': {
            1: 'very closely',
            2: 'fairly closely',
            3: 'not too closely',
            4: 'not at all closely',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 3,
            2: 2,
            3: 1,
            4: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'ordinal'
    },
    'q8': {
        'full':
        'Q8. In general, would you say you are satisfied or not satisfied with your choices of candidates in the election for governor on November 6th?',
        'short': 'satisfy_governor_choice',
        'use': 1,
        'code_text': {
            1: 'satisfied',
            2: 'not satisfied',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q9': {
        'full':
        'Q9. If the November 6th election for the U.S. Senate were being held today, would you vote for [ROTATE] [1] Kevin De Leon, a Democrat, [OR] [2] Dianne Feinstein, a Democrat?',
        'short': 'senate_choice',
        'use': 1,
        'code_text': {
            1: 'Kevin De Leon (DAY-LEON), a Democrat, California Senator',
            2: 'Dianne Feinstein (FINE-STINE), a Democrat, US Senator',
            3: 'neither/would not vote for U.S. Senator (volunteered)',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            3: 999,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q10': {
        'full':
        'Q10. In general, would you say you are satisfied or not satisfied with your choices of candidates in the election for US Senate on November 6th?',
        'short': 'satisfy_senate_choice',
        'use': 1,
        'code_text': {
            1: 'satisfied',
            2: 'not satisfied',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q11': {
        'full':
        'Q11. If the 2018 election for U.S. House of Representatives were being held today, would you vote for [ROTATE] [1] the Republican candidate [OR] [2] the Democratic candidate] in your district?',
        'short': 'house_choice',
        'use': 0,
        'code_text': {
            1: 'Rep/lean Rep',
            2: 'Dem/lean Dem',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 0,
            2: 1,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q12': {
        'full':
        "Q12. Which of the following is more important to you in candidates for US Congress' [ROTATE] (1) that they work with the Trump Administration [OR] (2) that they push back against the Trump Administration?",
        'short': 'congress_trump',
        'use': 1,
        'code_text': {
            1: 'work with the Trump Administration',
            2: 'push back against the Trump Administration',
            3: '(vol) both',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 0,
            2: 1,
            3: 999,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q13': {
        'full':
        'Q13. How enthusiastic would you say you are about voting for Congress this year - extremely enthusiastic, very enthusiastic, somewhat enthusiastic, not too enthusiastic, or not at all enthusiastic?',
        'short': 'enthus_congress_vote',
        'use': 1,
        'code_text': {
            1: 'extremely enthusiastic',
            2: 'very enthusiastic',
            3: 'somewhat enthusiastic',
            4: 'not too enthusiastic',
            5: 'not at all enthusiastic',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 4,
            2: 3,
            3: 2,
            4: 1,
            5: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'ordinal'
    },
    'q14': {
        'full':
        'Q14. Proposition 6 is the Eliminates Certain Road Repair and Transportation Funding. Requires Certain Fuel Taxes and Vehicle Fees be Approved by the Electorate. If held today, would you vote yes or no?',
        'short': 'opinion_proposition6',
        'use': 1,
        'code_text': {
            1: 'yes',
            2: 'no',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q15': {
        'full':
        'Q15. How important to you is the outcome of the vote on Proposition 6 - is it very important, somewhat important, not too important, or not at all important?',
        'short': 'importance_proposition6',
        'use': 1,
        'code_text': {
            1: 'very important',
            2: 'somewhat important',
            3: 'not too important',
            4: 'not at all important',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 3,
            2: 2,
            3: 1,
            4: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'ordinal'
    },
    'q16': {
        'full':
        'Q16. Proposition 10 is called the Expands Local Governments Authority to Enact Rent Control on Residential Property.  If the election were held today, would you vote yes or no on Proposition 10?',
        'short': 'opinion_proposition10',
        'use': 1,
        'code_text': {
            1: 'yes',
            2: 'no',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q17': {
        'full':
        'Q17. How important to you is the outcome of the vote on Proposition 10 - is it very important, somewhat important, not too important, or not at all important?',
        'short': 'imortance_proposition6',
        'use': 1,
        'code_text': {
            1: 'very important',
            2: 'somewhat important',
            3: 'not too important',
            4: 'not at all important',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 3,
            2: 2,
            3: 1,
            4: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'ordinal'
    },
    'q18': {
        'full':
        'Q18. Would you say traffic congestion on freeways and major roads is a big problem, somewhat of a problem, or not a problem in your region of California?',
        'short': 'opinion_traffic',
        'use': 1,
        'code_text': {
            1: 'big problem',
            2: 'somewhat of a problem',
            3: 'not a problem',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 2,
            2: 1,
            3: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'ordinal'
    },
    'q19': {
        'full':
        'Q19. How much of a problem is housing affordability in your part of California? Is it a big problem, somewhat of a problem, or not a problem?',
        'short': 'opinion_housing_afford',
        'use': 1,
        'code_text': {
            1: 'big problem',
            2: 'somewhat of a problem',
            3: 'not a problem',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 2,
            2: 1,
            3: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'ordinal'
    },
    'q20': {
        'full':
        'Q20. Overall, do you approve or disapprove of the way that Donald Trump is handling his job as president?',
        'short': 'approve_trump',
        'use': 1,
        'code_text': {
            1: 'approve',
            2: 'disapprove',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q21': {
        'full':
        'Q21. Overall, do you approve or disapprove of the way the U.S. Congress is handling its job?',
        'short': 'approve_congress',
        'use': 1,
        'code_text': {
            1: 'approve',
            2: 'disapprove',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q21a': {
        'full':
        "Q21a. Do you approve or disapprove of the U.S. Senate's vote to confirm Trump's nomination of Brett Kavanaugh to the U.S. Supreme Court?",
        'short': 'approve_kavanaugh',
        'use': 1,
        'code_text': {
            1: 'approve',
            2: 'disapprove',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q22': {
        'full':
        'Q22. Do you think things in the United States are generally going in the right direction or the wrong direction?',
        'short': 'opinion_us',
        'use': 1,
        'code_text': {
            1: 'right direction',
            2: 'wrong direction',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q23': {
        'full':
        'Q23. Turning to economic conditions, do you think that during the next 12 months the United States will have good times financially or bad times?',
        'short': 'opinion_us_econ',
        'use': 1,
        'code_text': {
            1: 'good times',
            2: 'bad times',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q24': {
        'full':
        'Q24. If you had to choose, would you rather have a smaller government providing fewer services, or a bigger government providing more services?',
        'short': 'opinion_goverment_size',
        'use': 1,
        'code_text': {
            1: 'smaller government, fewer services',
            2: 'bigger government, more services',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 0,
            2: 1,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q25': {
        'full':
        'Q25. In general, do you think laws covering the sale of guns should be more strict, less strict, or kept as they are now?',
        'short': 'opinion_gun_restriction',
        'use': 1,
        'code_text': {
            1: 'more strict',
            2: 'less strict',
            3: 'kept as they are now',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: -1,
            3: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'ordinal'
    },
    'q26': {
        'full':
        'Q26. All in all, would you favor or oppose building a wall along the entire border with Mexico?',
        'short': 'opinion_mexico_wall',
        'use': 1,
        'code_text': {
            1: 'favor',
            2: 'oppose',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q27': {
        'full':
        'Q27. Do you favor or oppose the California state and local governments making their own policies and taking actions, separate from the federal government, to protect the legal rights of undocumented immigrants in California?',
        'short': 'opinion_cal_own_policy',
        'use': 1,
        'code_text': {
            1: 'favor',
            2: 'oppose',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q28': {
        'full':
        'Q28. As you may know, a health reform bill was signed into law in 2010, known commonly as the Affordable Care Act or Obamacare. Given what you know about the health reform law, do you have a generally favorable/unfavorable opinion of it?',
        'short': 'opinion_obamacare',
        'use': 1,
        'code_text': {
            1: 'generally favorable',
            2: 'generally unfavorable',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q29': {
        'full':
        'Q29. Do you think it is the responsibility of the federal government to make sure all Americans have health care coverage, or is that NOT the responsibility of the federal government?',
        'short': 'opinion_federal_stance_healthcare',
        'use': 1,
        'code_text': {
            1: 'is responsibility of federal government [ASK Q29a]',
            2: 'is not responsibility of federal government [SKIP TO Q30]',
            8: '(vol) don’t know [SKIP TO 30]',
            9: '(vol) refuse [SKIP TO 30]'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q29a': {
        'full':
        'Q29a. Should health insurance [ROTATE 1&2] 1. (Be provided through a single national health insurance system run by the government) OR 2. (Continue to be provided through a mix of private insurance companies and government programs)?',
        'short': 'opinion_healthcare',
        'use': 1,
        'code_text': {
            1:
            'should be provided through a single national health insurance system run by the government',
            2:
            'should continue to be provided through a mix of private insurance companies and government programs',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q30': {
        'full':
        'Q30. Do you have a favorable or an unfavorable impression of the Democratic Party?',
        'short': 'view_democrat',
        'use': 1,
        'code_text': {
            1: 'favorable',
            2: 'unfavorable',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q31': {
        'full':
        'Q31. Do you have a favorable or an unfavorable impression of the Republican Party?',
        'short': 'view_republican',
        'use': 1,
        'code_text': {
            1: 'favorable',
            2: 'unfavorable',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q32': {
        'full':
        'Q32. In your view, do the Republican and Democratic parties do an adequate job representing the American people, or do they do such a poor job that a third major party is needed?',
        'short': 'view_main_parties',
        'use': 1,
        'code_text': {
            1: 'adequate job',
            2: 'third party is needed',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q33': {
        'full':
        'Q33. Next, would you consider yourself to be politically: Liberal or conservative?',
        'short': 'ideology',
        'use': 1,
        'code_text': {
            1: 'very liberal',
            2: 'somewhat liberal',
            3: 'middle-of-the-road',
            4: 'somewhat conservative',
            5: 'very conservative',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 2,
            2: 1,
            3: 0,
            4: -1,
            5: -2,
            8: 999,
            9: 999
        }[a],
        'type': 'ordinal'
    },
    'q34': {
        'full':
        'Q34. Generally speaking, how much interest would you say you have in politics - a great deal, a fair amount, only a little, or none?',
        'short': 'interest_politics',
        'use': 1,
        'code_text': {
            1: 'great deal',
            2: 'fair amount',
            3: 'only a little',
            4: 'none',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 3,
            2: 2,
            3: 1,
            4: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'ordinal'
    },
    'q35': {
        'full':
        'Q35. Thinking about the November 6th election, are you more enthusiastic about voting than usual, or less enthusiastic?',
        'short': 'enthus_nov_vote',
        'use': 1,
        'code_text': {
            1: 'more enthusiastic',
            2: 'less enthusiastic',
            3: '(vol) same/neither',
            4: '(vol) can’t vote',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            3: 999,
            4: 999,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q36': {
        'full':
        'Q36. How often would you say you vote - always, nearly always, part of the time, seldom, or never?',
        'short': 'freq_vote',
        'use': 1,
        'code_text': {
            1: 'always',
            2: 'nearly always',
            3: 'part of the time',
            4: 'seldom',
            5: 'never',
            8: '(vol) don’t know',
            9: '(vol) refuse '
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 4,
            2: 3,
            4: 2,
            3: 1,
            5: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'ordinal'
    },
    'q37': {
        'full':
        'Q37. And do you plan to vote in the statewide general election on November 6th? [IF RESPONSENT SAYS THEY WILL VOTE EARLY/VOTE ABSENTEE CODE THEM AS <PUNCH 1>] [IF SOMEONE SAYS THEY ALREADY VOTED CODE THEM AS <PUNCH 1>]',
        'short': 'plan_nov_vote',
        'use': 0,
        'code_text': {
            1: 'yes [ASK Q37a]',
            2: 'no [SKIP TO Q38]',
            8: '(vol) don’t know [SKIP TO Q38]',
            9: '(vol) refuse [SKIP TO Q38]'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q37a': {
        'full':
        'Q37a. Do you plan to vote at your local polling place, by mail ballot, or have you already voted? [INTERVIEWER: IF RESPONDENT SAYS THEY ALREADY VOTED BY MAIL/ABSENTEE BALLOT, CODE AS PUNCH <3> already voted]',
        'short':
        'plan_vote_method',
        'use':
        1,
        'code_text': {
            1: 'local polling place [SKIP TO Q38]',
            2: 'mail ballot [SKIP TO Q38]',
            3: 'already voted [ASK Q37b]',
            8: '(vol) don’t know [SKIP TO Q38]',
            9: '(vol) refuse [SKIP TO Q38]'
        },
        'code_num':
        lambda a: 999 if np.isnan(a) or int(a) == 8 or int(a) == 9 else int(a),
        'type':
        'categorical'
    },
    'q37b': {
        'full': 'Q37b. Did you vote in person or by mail?',
        'short': 'used_vote_method',
        'use': 1,
        'code_text': {
            1: 'in person',
            2: 'by mail',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 0,
            2: 1,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q38': {
        'full':
        'Q38. Regardless of how you may be registered, in politics today, do you consider yourself a Republican, Democrat or Independent?',
        'short':
        'party2',
        'use':
        0,
        'code_text': {
            1: 'Republican [SKIP TO Q38b]',
            2: 'Democrat [ASK Q38a]',
            3: 'Independent [SKIP TO Q38c]',
            8: '(vol) don’t know [SKIP TO Q38c]',
            9: '(vol) refuse [SKIP TO Q38c]'
        },
        'code_num':
        lambda a: 999 if np.isnan(a) or int(a) == 8 or int(a) == 9 else int(a),
        'type':
        'categorical'
    },
    'q38a': {
        'full':
        'Q38a. Do you consider yourself a strong Democrat or not a strong Democrat?',
        'short': 'strong_democrat2',
        'use': 0,
        'code_text': {
            1: 'strong',
            2: 'not very strong',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q38b': {
        'full':
        'Q38b. Do you consider yourself a strong Republican or not a strong Republican?',
        'short': 'strong_republican2',
        'use': 0,
        'code_text': {
            1: 'strong',
            2: 'not very strong',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'q38c': {
        'full':
        'Q38c. As of today do you lean more to the Republican Party or more to the Democratic Party?',
        'short': 'close_polotical_party2',
        'use': 0,
        'code_text': {
            1: 'Republican Party',
            2: 'Democratic Party',
            3: '(vol) neither',
            8: '(vol) don’t know',
            9: '(vol) refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 0,
            2: 1,
            3: 999,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'd1': {
        'full':
        'D1. Finally, we have a few demographic questions. What is your age?',
        'short':
        'age',
        'use':
        0,
        'code_text':
        lambda a: 'refuse (volunteered) [ASK D1a]'
        if np.isnan(a) or int(a) == 99 else a,
        'code_num':
        lambda a: 999 if np.isnan(a) or int(a) == 99 else a,
        'type':
        'numerical'
    },
    'd2': {
        'full': 'D2. Do you own or rent your current residence?',
        'short': 'residence_type',
        'use': 0,
        'code_text': {
            1: 'own',
            2: 'rent',
            3: 'neither (volunteered)',
            9: 'don’t know (volunteered)/refuse '
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            3: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'd3a': {
        'full':
        'D3a. Could you please tell me if you have lived at your current address for fewer than five years, five years to under 10 years, 10 years to under 20 years, or 20 years or more?',
        'short': 'year_same_place',
        'use': 1,
        'code_text': {
            1: 'fewer than five years',
            2: 'five years to under 10 years',
            3: '10 years to under 20 years',
            4: '20 years or more',
            5: 'all my life (volunteered)',
            9: 'refuse (volunteered)'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 4,
            9: 999
        }[a],
        'type': 'ordinal'
    },
    'd4': {
        'full':
        'D4. Are you a parent, stepparent, or legal guardian of any children 18 or under?',
        'short': 'has_children',
        'use': 0,
        'code_text': {
            1: 'yes',
            2: 'no',
            9: 'don’t know/refuse (volunteered)'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            9: 999
        }[a],
        'type': 'binary'
    },
    'd5': {
        'full':
        'Work Type',
        'short':
        'work_type',
        'use':
        0,
        'code_text': {
            1: 'full-time employed',
            2: 'part-time employed',
            3: 'not-employed',
            5: 'retired',
            8: 'disabled/on disability (volunteered)',
            9: 'don’t know/refuse (volunteered)'
        },
        'code_num':
        lambda a: 999 if np.isnan(a) or int(a) == 8 or int(a) == 9 else int(a),
        'type':
        'categorical'
    },
    'd6': {
        'full':
        'D6. Are you, yourself, now covered by any form of health insurance or health plan or do you not have health insurance at this time?',
        'short': 'has_health_insurance',
        'use': 1,
        'code_text': {
            1: 'yes, have insurance [ASK D6a]',
            2: 'no, do not have insurance [SKIP TO D6com]',
            9: 'refuse (volunteered) [SKIP TO D6com]'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            9: 999
        }[a],
        'type': 'binary'
    },
    'd6a': {
        'full':
        'D6a. Which of the following is your MAIN source of health insurance coverage?',
        'short':
        'health_insurance_type',
        'use':
        0,
        'code_text': {
            1: 'plan through your employer',
            2: 'plan through your spouse’s employer',
            3:
            'plan you purchased yourself (including Covered California, healthcare.gov, Obamacare)',
            4: 'Medicare',
            5: 'Medi-CAL (Medicaid, Healthy Families)',
            7: 'somewhere else [SPECIFY]',
            8: '(vol) plan through your parents/mother/father',
            9: '(vol) don’t know/refused'
        },
        'code_num':
        lambda a: 999 if np.isnan(a) or int(a) == 8 or int(a) == 9 else int(a),
        'type':
        'categorical'
    },
    'd7': {
        'full': 'D7. What was the last grade of school that you completed?',
        'short': 'education',
        'use': 1,
        'code_text': {
            1: 'some high school or less',
            2: 'high school graduate/GED',
            3: 'some college',
            4: 'college graduate',
            5: 'post graduate',
            6: 'trade school (volunteered)',
            9: 'refuse (volunteered)'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 4,
            6: 999,
            9: 999
        }[a],
        'type': 'ordinal'
    },
    'd8': {
        'full': 'D8. Are you of Hispanic, Latino or Spanish origin?',
        'short': 'is_hispanic',
        'use': 0,
        'code_text': {
            1: 'yes, Hispanic',
            2: 'no, NOT Hispanic',
            9: 'don’t know/refused (volunteered)'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            9: 999
        }[a],
        'type': 'binary'
    },
    'qd8a_1': {
        'full':
        "D8a. [For classification purposes, we'd like to know what your racial background is. Are you ... (MENTION #1)",
        'short': 'race1',
        'use': 0,
        'code_text': {
            1: 'white',
            2: 'black/African-American',
            3: 'Asian',
            4: 'Pacific Islander/Native Hawaiian',
            5: 'American Indian/Alaskan Native',
            6: 'Hispanic/Latino (volunteered)',
            7: 'other (specify)',
            9: 'don’t know/refused (volunteered)'
        },
        'code_num': lambda a: 999 if np.isnan(a) or int(a) == 9 else int(a),
        'type': 'categorical'
    },
    'qd8a_2': {
        'full':
        "D8a. [For classification purposes, we'd like to know what your racial background is. Are you ... (MENTION #2)",
        'short': 'race2',
        'use': 0,
        'code_text': {
            1: 'white',
            2: 'black/African-American',
            3: 'Asian',
            4: 'Pacific Islander/Native Hawaiian',
            5: 'American Indian/Alaskan Native',
            6: 'Hispanic/Latino (volunteered)',
            7: 'other (specify)',
            9: 'don’t know/refused (volunteered)'
        },
        'code_num': lambda a: 999 if np.isnan(a) or int(a) == 9 else int(a),
        'type': 'categorical'
    },
    'd8com': {
        'full': 'Race/Ethnicity',
        'short': 'race_ethnicity',
        'use': 0,
        'code_text': {
            1: 'Asian, non-Hispanic',
            2: 'black, non-Hispanic',
            3: 'Hispanic or Latino, any race',
            4: 'white, non-Hispanic',
            5: 'other, non-Hispanic',
            6: 'multi-race, non-Hispanic',
            9: 'refuse'
        },
        'code_num': lambda a: 999 if np.isnan(a) or int(a) == 9 else int(a),
        'type': 'categorical'
    },
    'd9': {
        'full': 'D9. Were you born in the United States?',
        'short': 'born_us',
        'use': 0,
        'code_text': {
            1: 'yes [SKIP TO D10]',
            2: 'no [ASK D9a]',
            9: 'don’t know/refuse (volunteered) [ASK D9a]'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            9: 999
        }[a],
        'type': 'binary'
    },
    'd9a': {
        'full': 'US Native',
        'short': 'citizen_us',
        'use': 1,
        'code_text': {
            1: 'yes, U.S. citizen',
            2: 'no, not a U.S. citizen',
            9: 'refuse (volunteered)'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            3: 999,  # there is no info for 3 in the codebook
            9: 999
        }[a],
        'type': 'binary'
    },
    'd11': {
        'full': 'Income',
        'short': 'income',
        'use': 1,
        'code_text': {
            1: 'under $20,000',
            2: '$20,000 to under $40,000',
            3: '$40,000 to under $60,000',
            4: '$60,000 to under $80,000',
            5: '$80,000 to under $100,000',
            6: '$100,000 to under $200,000',
            7: '$200,000 or more',
            9: 'don’t know/refuse (volunteered)'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 4,
            6: 5,
            7: 6,
            9: 999
        }[a],
        'type': 'ordinal'
    },
    'd12': {
        'full':
        'D12. Including yourself, how many adults 18 years of age or older live in your household?',
        'short': 'num_adults',
        'use': 0,
        'code_text': {
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 6,
            7: 7,
            8: 8,
            9: 9,
            10: 10,
            11: 11,
            12: 12,
            99: 'don’t know/refuse (volunteered)'
        },
        'code_num': lambda a: 999 if np.isnan(a) or int(a) == 99 else int(a),
        'type': 'numerical'
    },
    'd13a': {
        'full':
        'D13a. Does anyone in your household have a working cell phone?',
        'short': 'has_cellphone',
        'use': 0,
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            9: 999
        }[a],
        'type': 'binary'
    },
    'd14': {
        'full':
        'D14. Now thinking about your telephone use, is there at least one telephone INSIDE your home that is currently working and is not a cell phone?',
        'short': 'has_housephone',
        'use': 0,
        'code_text': {
            1: 'yes, have cell phone [SKIP TO D15]',
            2: 'no, do not [ASK D13a]',
            9: 'don’t know/refuse (volunteered) [ASK D13a]'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            9: 999
        }[a],
        'type': 'binary'
    },
    'd15': {
        'full':
        'D15. Finally, would you be willing to talk about these questions with a reporter from a news organization in your region for a story about this survey?',
        'short': 'talk_with_reporter',
        'use': 0,
        'code_text': {
            1: 'yes, someone in household has cell phone',
            2: 'no',
            9: '(vol) don’t know/refused'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0,
            9: 999
        }[a],
        'type': 'binary'
    },
    'd16': {
        'full':
        "D16. That's the end of the survey. We'd like to send you $5 for your time. Is there a mailing address where we can send you the money?",
        'short': 'recieve_money',
        'use': 0,
        'code_text': {
            1: '[gave mailing address]',
            2: 'respondent does not want the money (volunteered)'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 0
        }[a],
        'type': 'binary'
    },
    'lgdr': {
        'full': 'INTERVIEWER: RECORD YOUR GENDER (NOT GENDER OF RESPONDENT)',
        'short': 'interviewer_gender',
        'use': 0,
        'code_text': {
            1: 'Male',
            2: 'Female',
            3: '(vol) other',
            8: '(vol) don’t know',
            9: '(vol) refused'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 0,
            2: 1,
            3: 999,
            8: 999,
            9: 999
        }[a],
        'type': 'binary'
    },
    'd1a': {
        'full': 'D1A. Could you please tell me if you are between the ages of',
        'short': 'age_range',
        'use': 1,
        'code_text': {
            1: '18 to 24',
            2: '25 to 34',
            3: '35 to 44',
            4: '45 to 54',
            5: '55 to 64',
            6: '65 or older',
            9: 'refuse (volunteered)'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 4,
            6: 5,
            9: 999
        }[a],
        'type': 'ordinal'
    },
    'gender': {
        'full': 'Gender',
        'short': 'gender',
        'use': 0,
        'code_text': {
            1: 'Male',
            2: 'Female',
            3: '(vol) other',
            8: '(vol) don’t know',
            9: '(vol) refused'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            1: 1,
            2: 2,
            3: 3,
            8: 999,
            9: 999
        }[a],
        'type': 'categorical',
    },
    'language': {
        'full': 'Language',
        'short': 'language',
        'use': 0,
        'code_text': {
            0: 'English',
            1: 'Spanish'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            0: 0,
            1: 1
        }[a],
        'type': 'binary'
    },
    'likevote': {
        'full': 'Likely Voters',
        'short': 'likely_voters',
        'use': 0,
        'code_text': {
            0: 'Respondent not likely to vote',
            1: 'Respondent likely to vote'
        },
        'code_num': lambda a: 999 if np.isnan(a) else {
            0: 0,
            1: 1
        }[a],
        'type': 'binary'
    },
    'weight': {
        'full': 'Final adjusted weight',
        'short': 'adjusted_weight',
        'use': 0,
        'code_text': lambda a: a,
        'code_num': lambda a: a,
        'type': 'numerical'
    }
}
