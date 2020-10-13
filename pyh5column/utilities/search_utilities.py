import tables as tb


def _build_search_string(match):
    # example string matching for unicode, where the nodes are stored in a list
    # tb.Expr('(x=={0})|(x=={1})'.format('£'.encode('utf-8'), b'rst'), uservars={'x': nodes[2]}).eval()
    # build search expression
    uservars = {}
    nodesfound = {}
    match_strings = []

    # check for single match condition and convert to iterator
    if isinstance(match[0], str):
        match = [match]

    for cond in match:

        if isinstance(cond[0], str) or (len(cond) == 1):

            # check for single tuple
            if len(cond) == 1:
                cond = cond[0]

            col_name = cond[0]
            oper = cond[1]
            val = cond[2]

            if col_name not in nodesfound:
                lenvars = len(uservars)
                nname = 'n{lenvars}'.format(lenvars=lenvars)
                nodesfound[col_name] = nname
                uservars[nname] = col_name
            else:
                nname = nodesfound[col_name]

            if isinstance(val, str):
                val = val.encode('utf-8')
            match_strings.append(f'({nname}{oper}{val})')

        else:

            or_strings = []
            for col_name, oper, val in cond:

                if col_name not in nodesfound:
                    lenvars = len(uservars)
                    nname = 'n{lenvars}'.format(lenvars=lenvars)
                    nodesfound[col_name] = nname
                    uservars[nname] = col_name
                else:
                    nname = nodesfound[col_name]

                if isinstance(val, str):
                    val = val.encode('utf-8')
                or_strings.append(f'({nname}{oper}{val})')

            # add all elements or'd together with surrounding parenthesis
            match_strings.append(f"({'|'.join(or_strings)})")

    # join outer and condition
    return '&'.join(match_strings), uservars


def _filter_inds(col_dict, query):

    match_string, uservars = _build_search_string(query)
    # link user vars to existing columns in memory
    for name in list(uservars.keys()):
        uservars[name] = col_dict[uservars[name]]

    # run filtering and subselect columns
    inds = tb.Expr(match_string, uservars=uservars).eval()

    return inds