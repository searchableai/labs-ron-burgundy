# Retrieve NPR archive urls
years="2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019"
months="1 2 3 4 5 6 7 8 9 10 11 12"
days="01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31"
for y in $years; do
  for m in $months; do
    echo "checkpoint" $y $m
    for d in $days; do
      curl "https://www.npr.org/sections/politics/archive?date=$m-$d-$y" | grep "h2 class=\"title\"" >> npr_response.txt
      curl "https://www.npr.org/sections/business/archive?date=$m-$d-$y" | grep "h2 class=\"title\"" >> npr_response.txt
      curl "https://www.npr.org/sections/health/archive?date=$m-$d-$y" | grep "h2 class=\"title\"" >> npr_response.txt
      curl "https://www.npr.org/sections/world/archive?date=$m-$d-$y" | grep "h2 class=\"title\"" >> npr_response.txt
      curl "https://www.npr.org/sections/science/archive?date=$m-$d-$y" | grep "h2 class=\"title\"" >> npr_response.txt
      curl "https://www.npr.org/sections/national/archive?date=$m-$d-$y" | grep "h2 class=\"title\"" >> npr_response.txt
    done
    echo "checkpoint" $y $m >> npr_response.txt
  done
done
