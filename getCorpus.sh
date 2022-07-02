#!/bin/bash
psql -U scottalexandermalec -d causalehr -c "select pmid, predicate, subject_cui, subject_name, subject_type, object_cui, object_name, object_type from mro;"
