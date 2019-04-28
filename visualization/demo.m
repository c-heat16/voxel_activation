expt.description = 'science';
expt.data_path = 'data';
expt.subject = 'P1';

expt

load( sprintf('%s/data-%s-%s.mat', ...
			expt.data_path, expt.description, expt.subject) );

col = [12185 12190]

coord = meta.colToCoord(col,:)

mni = get_mni_coordinates(coord)

aal = get_aal_labels(mni);

for i = 1:length(aal),
	aal{i}
end
