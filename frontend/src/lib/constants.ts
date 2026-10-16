import { WasteClass, RecyclingGuidance } from './types';

export const WASTE_CLASSES: WasteClass[] = [
  'cardboard',
  'glass',
  'metal',
  'paper',
  'plastic',
  'trash'
];

export const CATEGORY_META: Record<
  WasteClass,
  {
    name: string;
    label: string;
    color: string;
    bgColor: string;
    borderColor: string;
    textColor: string;
    badgeClass: string;
    icon: string;
    description: string;
  }
> = {
  cardboard: {
    name: 'cardboard',
    label: 'Cardboard',
    color: '#d97706',
    bgColor: 'bg-amber-50',
    borderColor: 'border-amber-200',
    textColor: 'text-amber-800',
    badgeClass: 'bg-amber-100 text-amber-800 border-amber-300',
    icon: 'Package',
    description: 'Corrugated boxes, packaging cartons, and cereal containers.'
  },
  glass: {
    name: 'glass',
    label: 'Glass',
    color: '#0d9488',
    bgColor: 'bg-teal-50',
    borderColor: 'border-teal-200',
    textColor: 'text-teal-800',
    badgeClass: 'bg-teal-100 text-teal-800 border-teal-300',
    icon: 'Wine',
    description: 'Clean bottles and condiment jars. 100% infinitely recyclable.'
  },
  metal: {
    name: 'metal',
    label: 'Metal',
    color: '#475569',
    bgColor: 'bg-slate-100',
    borderColor: 'border-slate-300',
    textColor: 'text-slate-800',
    badgeClass: 'bg-slate-200 text-slate-800 border-slate-300',
    icon: 'Layers',
    description: 'Aluminum drink cans, steel food tins, and clean baking foil.'
  },
  paper: {
    name: 'paper',
    label: 'Paper',
    color: '#0284c7',
    bgColor: 'bg-sky-50',
    borderColor: 'border-sky-200',
    textColor: 'text-sky-800',
    badgeClass: 'bg-sky-100 text-sky-800 border-sky-300',
    icon: 'FileText',
    description: 'Office sheets, clean newspapers, notebooks, and magazines.'
  },
  plastic: {
    name: 'plastic',
    label: 'Plastic',
    color: '#10b981',
    bgColor: 'bg-emerald-50',
    borderColor: 'border-emerald-200',
    textColor: 'text-emerald-800',
    badgeClass: 'bg-emerald-100 text-emerald-800 border-emerald-300',
    icon: 'Milk',
    description: 'Rigid containers, PET beverage bottles (#1), and milk jugs (#2).'
  },
  trash: {
    name: 'trash',
    label: 'Trash (Non-Recyclable)',
    color: '#e11d48',
    bgColor: 'bg-rose-50',
    borderColor: 'border-rose-200',
    textColor: 'text-rose-800',
    badgeClass: 'bg-rose-100 text-rose-800 border-rose-300',
    icon: 'Trash2',
    description: 'Contaminated waste, multi-layer pouches, and general municipal refuse.'
  }
};

export const STATIC_RECYCLING_GUIDES: Record<WasteClass, RecyclingGuidance> = {
  cardboard: {
    category: 'cardboard',
    is_recyclable: true,
    summary: 'Cardboard and corrugated boxes are highly recyclable when clean and dry.',
    common_examples: ['Shipping boxes', 'Cereal boxes', 'Shoe boxes', 'Clean egg cartons', 'Cardboard tubes'],
    disposal_guide: 'Flatten all boxes to save space in the recycling bin. Remove plastic packing tape, styrofoam inserts, and shipping wrap before disposal.',
    recycling_tips: [
      'Flatten boxes completely before putting them into recycling bins.',
      'Remove all plastic bubble wrap and polystyrene packing peanuts.',
      'Keep boxes dry; wet cardboard fibers break down too quickly for reprocessing.'
    ],
    contamination_warnings: [
      'Pizza boxes soaked with heavy grease or cheese cannot be recycled with cardboard.',
      'Wax-coated produce boxes ruin paper recycling batches.'
    ]
  },
  glass: {
    category: 'glass',
    is_recyclable: true,
    summary: 'Glass bottles and jars are 100% recyclable infinitely without loss of purity or quality.',
    common_examples: ['Beverage bottles', 'Jam jars', 'Sauce containers', 'Glass perfume bottles'],
    disposal_guide: 'Rinse containers with water to remove food residues. Check if your municipal service accepts metal lids attached or separated.',
    recycling_tips: [
      'A simple rinse is sufficient; there is no need to run through a dishwasher.',
      'Labels will burn away during the industrial remelting process.',
      'Separate by color (clear, green, amber) if your local recycling drop-off requires sorting.'
    ],
    contamination_warnings: [
      'Do NOT dispose of window panes, mirrors, ceramics, drinking tumblers, or Pyrex in container glass bins.',
      'Treated glassware melts at different temperatures and ruins furnace batches.'
    ]
  },
  metal: {
    category: 'metal',
    is_recyclable: true,
    summary: 'Aluminum and steel cans are among the most energy-efficient materials to recycle.',
    common_examples: ['Soda cans', 'Food tins & soup cans', 'Clean aluminum trays', 'Aerosol cans (empty)'],
    disposal_guide: 'Empty and lightly rinse to prevent odors and pests. Place metal lids inside the cans or squeeze the rim slightly.',
    recycling_tips: [
      'Crushing aluminum beverage cans saves space in household bins.',
      'Recycling aluminum saves 95% of the energy needed to refine new metal from bauxite ore.',
      'Ensure aerosol cans are completely depressurized.'
    ],
    contamination_warnings: [
      'Paint cans, motor oil canisters, and toxic chemical bottles belong at hazardous waste drop-off centers.',
      'Foil caked with heavy burnt food must be discarded in the trash.'
    ]
  },
  paper: {
    category: 'paper',
    is_recyclable: true,
    summary: 'Clean office paper, newspapers, and brochures can be pulped into new paper products.',
    common_examples: ['Office printing paper', 'Newspapers & magazines', 'Notebook paper', 'Brochures & envelopes'],
    disposal_guide: 'Keep paper dry and unsoiled. Place directly into the dry recycling receptacle.',
    recycling_tips: [
      'Avoid shredding paper unless necessary for confidential data; shredding cuts fibers short.',
      'Standard staples and paper clips are removed magnetically at pulping mills.',
      'Separate from kitchen waste.'
    ],
    contamination_warnings: [
      'Used paper towels, napkins, facial tissues, and greasy wrappers are NOT recyclable.',
      'Thermal cash register receipts contain BPA/BPS and belong in the trash.'
    ]
  },
  plastic: {
    category: 'plastic',
    is_recyclable: true,
    summary: 'Rigid plastic containers (#1 PET, #2 HDPE, #5 PP) are widely accepted by curbside programs.',
    common_examples: ['Water and soda bottles (#1)', 'Milk jugs and shampoo bottles (#2)', 'Yogurt tubs (#5)'],
    disposal_guide: 'Empty all liquids, rinse out residue, and firmly screw caps back on.',
    recycling_tips: [
      'Check the resin identification triangle on the container bottom.',
      'Bottles with necks (#1 and #2) have the highest global recycling recovery rates.',
      'Keep screw caps on bottles so they do not fall through sorting grates.'
    ],
    contamination_warnings: [
      'Plastic bags, bubble wrap, chip bags, and shrink wrap tangle mechanical sorting equipment; return these to retail grocery drop-offs.',
      'Black plastic food trays are invisible to optical sorting sensors and end up in landfill.'
    ]
  },
  trash: {
    category: 'trash',
    is_recyclable: false,
    summary: 'Non-recyclable municipal solid waste, composite packaging, and heavily contaminated materials.',
    common_examples: ['Greasy food wrappers', 'Broken ceramics', 'Styrofoam cups', 'Hygiene items', 'Multi-layer chip bags'],
    disposal_guide: 'Place in designated municipal trash bags and dispose into general landfill bins.',
    recycling_tips: [
      'Minimize landfill waste by opting for reusables (stainless bottles, canvas bags).',
      'Compost organic kitchen scraps separately where municipal composting is provided.',
      'Avoid "wish-cycling" non-recyclables into the blue bin, as it can spoil clean batches.'
    ],
    contamination_warnings: [
      'Throwing hazardous electronics, lithium batteries, or paint into household trash causes landfill fires.'
    ]
  }
};

export const CREATOR_INFO = {
  name: 'Priya Sharma',
  degree: 'B.Tech, Computer Science and Engineering',
  role: 'Full-Stack Developer & Machine Learning Engineer',
  project: 'RecycleVision AI',
  tagline: 'Smart Waste Classification & Recycling Analytics',
  institution: 'Engineering University'
};
