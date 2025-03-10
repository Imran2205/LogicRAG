from kanren import Relation, facts, run, var, eq, conde
from kanren.core import lall
import logging
import time
from typing import List, Dict, Set
import re


class KanrenKB:
    def __init__(self, logger):
        self.type_of = Relation('TypeOf')
        self.color_of = Relation('ColorOf')
        self.railtrack = Relation('Railtrack')
        self.rules = []
        self.antecedent_predicates = []
        self.consequent_predicates = []
        self.logger = logger

        self.relations = {
            'TypeOf': self.type_of,
            'ColorOf': self.color_of,
            'Railtrack': self.railtrack
        }

    def _extract_predicates_from_kb(self, kb_filename: str) -> Set[str]:
        """Extract unique predicates from all lines in KB file"""
        predicates = set()

        with open(kb_filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if '==>' in line:
                    # Handle rules: split and process both sides
                    antecedent, consequent = line.split('==>')
                    # Process both parts
                    parts = [antecedent.strip('()'), consequent.strip('()')]
                    for part in parts:
                        matches = re.findall(r'([A-Za-z][A-Za-z0-9_]*)\(', part)
                        predicates.update(matches)
                else:
                    # Handle facts and other lines
                    matches = re.findall(r'([A-Za-z][A-Za-z0-9_]*)\(', line)
                    predicates.update(matches)

        # Remove predicates that are already defined
        # predicates = predicates - set(self.relations.keys())

        self.logger.debug(f"Found {len(predicates)} additional unique predicates")
        self.logger.debug(f"Additional predicates: {sorted(predicates)}")
        return predicates

    def _initialize_relations(self, predicates: Set[str]):
        """Create Relation objects for each predicate"""
        for pred in predicates:
            self.relations[pred] = Relation(pred)
        self.logger.debug(f"Total relations initialized: {len(self.relations)}")
        self.logger.debug(f"Relations: {', '.join(sorted(self.relations.keys()))}")

    def load_kb(self, kb_filename: str, obj_info_filename: str):
        """Load knowledge base from files"""
        self.logger.debug("Extracting predicates and initializing relations...")

        # Extract predicates from both files
        predicates = self._extract_predicates_from_kb(kb_filename)
        obj_predicates = self._extract_predicates_from_kb(obj_info_filename)
        predicates.update(obj_predicates)

        # Initialize additional relations
        self._initialize_relations(predicates)

        self.logger.debug("Loading knowledge base...")
        start_time = time.time()

        # Load rules and facts from KB file
        with open(kb_filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith('('):
                    self._process_rule(line)
                else:
                    self._add_fact(line)

        # Load object properties
        with open(obj_info_filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    self._add_fact(line)
                    # print(line)

        self._add_rule()

        self.logger.debug(f"KB loaded in {time.time() - start_time:.2f} seconds")

    def _process_rule(self, rule_str: str):
        """Process an implication rule"""
        # Remove outer parentheses and split on ==>
        rule = rule_str.lstrip('(')[:-1]  # .strip('()')

        if '==>' in rule:
            antecedent, consequent = rule.split('==>', 1)

            # Clean and parse both parts
            def clean_and_parse(expr):
                # Remove all parentheses
                expr = expr.replace('((', '(').replace('))', ')')
                # Split on & and clean each predicate
                predicates = [p.strip() for p in expr.split('&')]
                # Remove empty strings
                predicates = [p for p in predicates if p]
                return predicates

            self.antecedent_predicates.append(clean_and_parse(antecedent))
            self.consequent_predicates.append(clean_and_parse(consequent))

            # Debug output
            # print("Rule:", rule_str)
            # print("Parsed predicates:")
            # print("Antecedent:", len(self.antecedent_predicates))
            # print("Consequent:", len(self.consequent_predicates))
            # print("-" * 50)

            # self._add_rule(antecedent_predicates, consequent_predicates)

    def _add_rule(self):
        """Add a rule to the knowledge base with implication logic"""
        x, y = var(), var()

        for pred_ind, antecedent_predicate in enumerate(self.antecedent_predicates):
            # Get results from antecedent predicates
            results = None

            # Process each antecedent predicate
            for pred in antecedent_predicate:
                pred_name = pred[:pred.index('(')]

                if pred_name not in self.relations:
                    self.relations[pred_name] = Relation(pred_name)

                args = pred[pred.index('(') + 1:pred.rindex(')')].split(',')
                args = [arg.strip() for arg in args]

                query_terms = []
                for arg in args:
                    if arg == 'x':
                        query_terms.append(x)
                    elif arg == 'y':
                        query_terms.append(y)
                    else:
                        query_terms.append(arg)

                # For single argument predicates
                if len(query_terms) == 1:
                    current_results = set(run(0, x, self.relations[pred_name](x)))
                else:  # For two argument predicates
                    current_results = set(run(0, (x, y), self.relations[pred_name](x, y)))

                # print(f"Results for {pred_name}: {current_results}")

                if results is None:
                    results = current_results
                else:
                    results = results.intersection(current_results)

            # Process consequent with the results
            consequent_predicate = self.consequent_predicates[pred_ind]
            for pred in consequent_predicate:
                pred_name = pred[:pred.index('(')]

                if pred_name not in self.relations:
                    self.relations[pred_name] = Relation(pred_name)

                args = pred[pred.index('(') + 1:pred.rindex(')')].split(',')
                args = [arg.strip() for arg in args]

                # print(f"Adding consequent facts for {pred_name} with results: {results}")

                # Add facts based on argument count
                if len(args) == 1:  # Single variable predicate
                    for result in results:
                        # print(result)
                        if isinstance(result, tuple):
                            facts(self.relations[pred_name], (result[0],))
                        else:
                            facts(self.relations[pred_name], (result,))

                    # print(f"After adding facts, query result for {pred_name}:",
                    #   list(run(0, x, self.relations[pred_name](x))))
                else:  # Two variable predicate
                    for result in results:
                        # print(result)
                        facts(self.relations[pred_name], (result[0], result[1]))
                        facts(self.relations[pred_name], (result[1], result[0]))

                    # print(f"After adding facts, query result for {pred_name}:",
                    #   list(run(0, (x, y), self.relations[pred_name](x, y))))

    def _add_fact(self, fact_str: str):
        """Add a single fact to the knowledge base"""
        try:
            pred_name = fact_str[:fact_str.index('(')]
            args = fact_str[fact_str.index('(') + 1:fact_str.rindex(')')].split(',')
            args = [arg.strip() for arg in args]

            if pred_name in self.relations:
                # print(f"**{pred_name}**", [tuple(args)])
                facts(self.relations[pred_name], tuple(args))
            else:
                self.logger.warning(f"Unknown predicate in fact: {pred_name}")
        except Exception as e:
            self.logger.error(f"Error adding fact '{fact_str}': {e}")

    def conjunctive_query(self, queries: List[str], var_name: str = 'x'):
        """Run a conjunctive query with multiple conditions"""
        # Create variables for x and y
        x, y = var(), var()
        main_var = x if var_name == 'x' else y

        # Process queries into goals
        goals = []
        infer_sets = []
        for query in queries:
            pred_name = query[:query.index('(')]
            args = query[query.index('(') + 1:query.rindex(')')].split(',')
            args = [arg.strip() for arg in args]

            # Skip if relation doesn't exist
            if pred_name not in self.relations:
                self.logger.warning(f"Predicate {pred_name} not found in relations")
                continue

            # Build query terms
            query_terms = []
            for arg in args:
                if arg == 'x':
                    query_terms.append(x)
                elif arg == 'y':
                    query_terms.append(y)
                else:
                    query_terms.append(arg)
            infer_sets.append(set(run(0, main_var, self.relations[pred_name](*query_terms))))
            goals.append(self.relations[pred_name](*query_terms))

        # Run the query with all conditions
        results = set.intersection(*infer_sets)

        # return list(results)

        # print(infer_sets)
        if goals:
            results = run(0, main_var, lall(*goals))
            # print(results)
            return list(results)
        return []

    def kb_query(self, queries: List[List[str]]):
        """Handle queries involving multiple objects and their relationships"""
        # Get results for first object (x)

        if len(queries[1]) == 0 or len(queries[2]) == 0:
            x_results = self.conjunctive_query(queries[0], 'x')
            self.logger.debug(f"Found {len(x_results)} matches for first object")
            return x_results

        x_results = self.conjunctive_query(queries[0], 'x')
        self.logger.debug(f"Found {len(x_results)} matches for first object")

        # Get results for second object (y)
        y_results = self.conjunctive_query(queries[1], 'y')
        self.logger.debug(f"Found {len(y_results)} matches for second object")

        if not x_results or not y_results:
            return []

        # Check relationship between objects
        final_results = []
        for x_obj in x_results:
            for y_obj in y_results:
                # For each relationship query
                all_relationships_match = True
                for relationship_query in queries[2]:
                    pred_name = relationship_query[:relationship_query.index('(')]
                    # print(pred_name)
                    if pred_name not in self.relations:
                        self.logger.warning(f"Relationship predicate {pred_name} not found")
                        continue

                    # Create a specific query for this pair of objects
                    v = var()  # dummy variable for run query
                    # print(pred_name, x_obj, y_obj)
                    result = run(1, v, self.relations[pred_name](x_obj, y_obj))
                    # print(result)

                    if not result:
                        all_relationships_match = False
                        break

                if all_relationships_match:
                    final_results.append((x_obj, y_obj))

        return final_results


def parse_query(query_str):
    """Convert query string to grouped lists of predicates"""
    # Split on ^ to get individual predicates
    predicates = query_str.split('^')

    # Clean predicates and group them
    x_predicates = []
    y_predicates = []
    xy_predicates = []

    for pred in predicates:
        pred = pred.strip()

        # Skip empty predicates
        if not pred:
            continue

        # Check predicate arguments
        args = pred[pred.index('(') + 1:pred.rindex(')')].split(',')
        args = [arg.strip() for arg in args]

        # Group based on variables present
        if 'x' in args and 'y' in args:
            xy_predicates.append(pred)
        elif 'x' in args:
            x_predicates.append(pred)
        elif 'y' in args:
            y_predicates.append(pred)

    return [x_predicates, y_predicates, xy_predicates]


def query_kb(kb_file, obj_info_file, queries, logger=None):
    if not logger:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

    kb = KanrenKB(logger)
    kb.load_kb(kb_file, obj_info_file)

    answers_ = []

    for query in queries:
        query_list = parse_query(query)

        results = kb.kb_query(query_list)

        # print(results)

        if len(results) > 0:
            answers_.append(1)
        else:
            answers_.append(0)

    return answers_
