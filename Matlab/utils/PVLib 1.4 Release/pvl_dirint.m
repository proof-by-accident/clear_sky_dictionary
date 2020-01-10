function dirintDNI = pvl_dirint(GHI, Z, doy, pressure, varargin)
% PVL_DIRINT Determine DNI from GHI using the DIRINT modification of the DISC model
%
% Syntax
%   dirintDNI = pvl_dirint(GHI, Z, doy, pressure)
%   dirintDNI = pvl_dirint(GHI, Z, doy, pressure, UseDelKtPrime)
%   dirintDNI = pvl_dirint(GHI, Z, doy, pressure, UseDelKtPrime, DewPtTemp)
%
% Description
%   Implements the modified DISC model known as "DIRINT" introduced in [1].
%   DIRINT predicts direct normal irradiance (DNI) from measured global
%   horizontal irradiance (GHI). DIRINT improves upon the DISC model by
%   using time-series GHI data and dew point temperature information. The
%   effectiveness of the DIRINT model improves with each piece of
%   information provided.
%
% Inputs:   
%   GHI - a scalar or vector of global horizontal irradiance in W/m^2. If GHI
%     is a vector it must be of the same size as all other vector inputs.
%     GHI must be >=0.
%   Z - a scalar or vector of true (not refraction-corrected) zenith
%     angles in decimal degrees. If Z is a vector it must be of the
%     same size as all other vector inputs. Z must be >=0 and <=180.
%   doy - a scalar or vector of values providing the day of the year. If
%     doy is a vector it must be of the same size as all other vector inputs.
%     doy must be >= 1 and < 367.
%   pressure - a scalar or vector of values providing the site pressure in
%     Pascal. If pressure is a vector it must be of the same size as all
%     other vector inputs. pressure must be >=0. Pressure may be measured
%     or an average pressure may be calculated from site altitude.
%   UseDelKtPrime - a numeric scalar indicating if the user would like to
%     utilize the time-series nature of the GHI measurements. A value of 0
%     will not use the time-series improvements, any other numeric value
%     will use time-series improvements. It is recommended that time-series
%     data only be used if the time between measured data points is less
%     than 1.5 hours. If UseDelKtPrime is not provided, the default is 1
%     (use time-series improvements). If none of the input arguments are
%     vectors, then time-series improvements are not used (because it's not
%     a time-series).
%   DewPtTemp - a scalar or vector of surface dew point temperatures, in 
%     degrees C. If DewPtTemp is a vector, it must be of the same size as
%     other vector inputs. Values of DewPtTemp may be numeric or NaN. Any
%     single time period point with a DewPtTemp=NaN does not have dew point
%     improvements applied. If DewPtTemp is not provided, then dew point 
%     improvements are not applied.  
%
% Output:   
%   dirintDNI - the modeled direct normal irradiance in W/m^2 provided by the
%     DIRINT model. dirintDNI is a column vector with the same number of
%     elements as the input vector(s).
%
% Sources
%
% [1]Perez, R., P. Ineichen, E. Maxwell, R. Seals and A. Zelenka, (1992).
%   "Dynamic Global-to-Direct Irradiance Conversion Models".  ASHRAE 
%   Transactions-Research Series, pp. 354-369
%
% [2] Maxwell, E. L., "A Quasi-Physical Model for Converting Hourly 
%   Global Horizontal to Direct Normal Insolation", Technical 
%   Report No. SERI/TR-215-3087, Golden, CO: Solar Energy Research 
%   Institute, 1987.
%
% DIRINT model requires time series data (ie. one of the inputs must be a
% vector of length >2.
%
% See also PVL_DATE2DOY PVL_EPHEMERIS PVL_ALT2PRES PVL_LOUCHE
% PVL_ORGILL_HOLLANDS PVL_REINDL_1 PVL_REINDL_2 PVL_ERBS PVL_DISC

maxvectorsize = max([numel(GHI) numel(Z) numel(doy) numel(pressure)]);

p = inputParser;
p.addRequired('GHI', @(x) all(isnumeric(x) & isvector(x) & x>=0));
p.addRequired('Z', @(x) (all(isnumeric(x) & x<=180 & x>=0 & isvector(x))));
p.addRequired('doy', @(x) (all(isnumeric(x) & isvector(x) & x>=1 & x<367)));
p.addRequired('pressure', @(x) all(isnumeric(x) & x>=0 & isvector(x)));
p.addOptional('UseDelKtPrime', 1, @(x) (isscalar(x) & isnumeric(x)));
p.addOptional('DewPtTemp', nan(maxvectorsize,1), ...
    @(x) all((isnan(x) | isnumeric(x)) & x>=-273 & isvector(x)));
p.parse(GHI,Z,doy, pressure, varargin{:});

GHI = p.Results.GHI(:);
Z=p.Results.Z(:);
doy=p.Results.doy(:);
pressure = p.Results.pressure(:);
Td = p.Results.DewPtTemp(:).*ones(maxvectorsize,1); % If DewPtTemp was scalar, you need to make it a vector
UseDelKtPrime = p.Results.UseDelKtPrime;
% Calculate Kt as in the DISC model, also get the DNI from the DISC model 
[DiscDNI, Kt] = pvl_disc(GHI, Z, doy, pressure);

% Initialize vectors to be column vectors of the same size as DiscDNI.
dirintDNI = zeros(numel(DiscDNI),1);
DIRINTCoeffs = zeros(numel(DiscDNI),1);
DelKtPrimeBin = zeros(numel(DiscDNI),1);
KtPrimeBin = zeros(numel(DiscDNI),1);
WBin = zeros(numel(DiscDNI),1);
ZBin = zeros(numel(DiscDNI),1);

% Absolute Airmass, per the DISC model
% Note that we calculate the AM pressure correction slightly differently
% than Perez. He uses altitude, we use pressure (which we calculate
% slightly differently)
AM = 1./(cosd(Z) + 0.15.*((93.885-Z).^(-1.253))) .* pressure./ 101325;

% Load the DIRINT coefficient lookup tables (see subfunction below)
DIRINTLookupTbl = getdirintcoeffs();

% Generate KtPrime and limit it to 0.82 as in the DIRINT code from SRRL
KtPrime = Kt./(1.031.*exp(-1.4./(0.9+9.4./AM))+0.1);
KtPrime(KtPrime > 0.82) = 0.82; % From SRRL code

NextKtPrime = zeros(numel(KtPrime),1);
LastKtPrime = zeros(numel(KtPrime),1);

% Create DelKtPrime. If we have chosen to utilize the time-series
% nature of data (i.e. UseDelKtPrime ~=0), then create a signal to use Bin
% 7. If we have time-series data (i.e. number of input elements >=2) and we
% have chosen to use it, generate DelKtPrime.
if (numel(KtPrime)>=2 && UseDelKtPrime)
    % Shift KtPrime to create NextKtPrime and LastKtPrime. This allows for
    % elementwise averaging.
    NextKtPrime(1:end-1) = KtPrime(2:end);
    LastKtPrime(2:end) = KtPrime(1:end-1);
    
    % For the points which do not have a next KtPrime (the last point) or a
    % previous KtPrime (the first point), fill NextKtPrime(lastpoint) with
    % LastKtPrime(lastpoint), and fill LastKtPrime(firstpoint) with
    % NextKtPrime(firstpoint). Then, when the averaging is performed, the
    % first/last values will only use the available KtPrime value
    % after/before them.   
    NextKtPrime(end) = LastKtPrime(end);
    LastKtPrime(1) = NextKtPrime(1);
    
    DelKtPrime = 0.5*(abs(KtPrime-NextKtPrime)+abs(KtPrime-LastKtPrime));
else
    DelKtPrime = -1; % This is a signaler to use Bin 7 
end

% Generate W from dew point temperature
W=exp(0.07.*Td - 0.075);
W(isnan(Td)) = -1; % A signal to use Bin 5

% Error check KtPrime and create KtPrime bins
if any(KtPrime > 1)
    error('Error in pvl_dirint, KtPrime values above 1 are not allowed.')
end
KtPrimeBin(KtPrime>=0 & KtPrime<0.24) = 1;
KtPrimeBin(KtPrime>=0.24 & KtPrime<0.4) = 2;
KtPrimeBin(KtPrime>=0.4 & KtPrime<0.56) = 3;
KtPrimeBin(KtPrime>=0.56 & KtPrime<0.7) = 4;
KtPrimeBin(KtPrime>=0.7 & KtPrime<0.8) = 5;
KtPrimeBin(KtPrime>=0.8 & KtPrime<=1) = 6;

% Create Zenith angle bins
ZBin(Z>=0 & Z<25) = 1;
ZBin(Z>=25 & Z<40) = 2;
ZBin(Z>=40 & Z<55) = 3;
ZBin(Z>=55 & Z<70) = 4;
ZBin(Z>=70 & Z<80) = 5;
% This bin (6) is extra large to accomodate valid values of Z, even if the sun
% is below the horizon. The DISC model should reduce the DNI to 0 (hence
% the DIRINT DNI should be 0) for values of Z>87.
ZBin(Z>=80 & Z<=180) = 6;

% Create the bins for W (based on dew point temperature)
WBin(W>=0 & W<1) = 1;
WBin(W>=1 & W<2) = 2;
WBin(W>=2 & W<3) = 3;
WBin(W>=3 & W<inf) = 4;
WBin(W == -1) = 5;

% Error check values of DelKtPrime, and create DelKtPrime binning.
if any(DelKtPrime > 1)
    error('Error in pvl_dirint, DelKtPrime values above 1 are not allowed.')
end
DelKtPrimeBin(DelKtPrime>=0 & DelKtPrime<0.015) = 1;
DelKtPrimeBin(DelKtPrime>=0.015 & DelKtPrime<0.035) = 2;
DelKtPrimeBin(DelKtPrime>=0.035 & DelKtPrime<0.07) = 3;
DelKtPrimeBin(DelKtPrime>=0.07 & DelKtPrime<0.15) = 4;
DelKtPrimeBin(DelKtPrime>=0.15 & DelKtPrime<0.3) = 5;
DelKtPrimeBin(DelKtPrime>=0.3 & DelKtPrime<=1) = 6;
DelKtPrimeBin(DelKtPrime == -1) = 7;

% Create the "Lookup4D" function to allow fast vector searches into a 4D
% table. This essentially creates a linear index based on the bin numbers
% and the size of the array, then indexes into the array.
Lookup4D = @(array,a,b,c,d) array(((d-1).*numel(array(:,:,:,1))+(c-1).*numel(array(:,:,1,1))+(b-1).*numel(array(:,1,1,1))+(a-1)+1));

% Use the Lookup4D function to find coefficients associated with each time
% step
DIRINTCoeffs = Lookup4D(DIRINTLookupTbl,KtPrimeBin, ZBin, DelKtPrimeBin, WBin);

% These may be unnecessary, as the current version of the DISC model will
% not give DNI values <0.
dirintDNI(DiscDNI > 0) = DiscDNI(DiscDNI>0) .* DIRINTCoeffs(DiscDNI>0);
dirintDNI(DiscDNI < 0) = 0;
end


function Coeffs = getdirintcoeffs()
%% Code to generate DIRINT 4-dimensional lookup table. 
% The correct input to look them up is as follows:
% CoefficientYouWant = Coeffs(KtPrimeBin, ZBin, DelKtPrimeBin, WBin)
% The most efficient way to look them up is to use a linear index into
% the 4-d table.
Coeffs(1,1,:,:)=     [...
    0.385230, 0.385230, 0.385230, 0.462880, 0.317440;...
    0.338390, 0.338390, 0.221270, 0.316730, 0.503650;...
    0.235680, 0.235680, 0.241280, 0.157830, 0.269440;...
    0.830130, 0.830130, 0.171970, 0.841070, 0.457370;...
    0.548010, 0.548010, 0.478000, 0.966880, 1.036370;...
    0.548010, 0.548010, 1.000000, 3.012370, 1.976540;...
    0.582690, 0.582690, 0.229720, 0.892710, 0.569950 ];

Coeffs(1,2,:,:)=     [...
    0.131280, 0.131280, 0.385460, 0.511070, 0.127940;...
    0.223710, 0.223710, 0.193560, 0.304560, 0.193940;...
    0.229970, 0.229970, 0.275020, 0.312730, 0.244610;...
    0.090100, 0.184580, 0.260500, 0.687480, 0.579440;...
    0.131530, 0.131530, 0.370190, 1.380350, 1.052270;...
    1.116250, 1.116250, 0.928030, 3.525490, 2.316920;...
    0.090100, 0.237000, 0.300040, 0.812470, 0.664970 ];

Coeffs(1,3,:,:)=     [...
    0.587510, 0.130000, 0.400000, 0.537210, 0.832490;...
    0.306210, 0.129830, 0.204460, 0.500000, 0.681640;...
    0.224020, 0.260620, 0.334080, 0.501040, 0.350470;...
    0.421540, 0.753970, 0.750660, 3.706840, 0.983790;...
    0.706680, 0.373530, 1.245670, 0.864860, 1.992630;...
    4.864400, 0.117390, 0.265180, 0.359180, 3.310820;...
    0.392080, 0.493290, 0.651560, 1.932780, 0.898730 ];

Coeffs(1,4,:,:)=     [...
    0.126970, 0.126970, 0.126970, 0.126970, 0.126970;...
    0.810820, 0.810820, 0.810820, 0.810820, 0.810820;...
    3.241680, 2.500000, 2.291440, 2.291440, 2.291440;...
    4.000000, 3.000000, 2.000000, 0.975430, 1.965570;...
    12.494170, 12.494170, 8.000000, 5.083520, 8.792390;...
    21.744240, 21.744240, 21.744240, 21.744240, 21.744240;...
    3.241680, 12.494170, 1.620760, 1.375250, 2.331620 ];

Coeffs(1,5,:,:)=     [...
    0.126970, 0.126970, 0.126970, 0.126970, 0.126970;...
    0.810820, 0.810820, 0.810820, 0.810820, 0.810820;...
    3.241680, 2.500000, 2.291440, 2.291440, 2.291440;...
    4.000000, 3.000000, 2.000000, 0.975430, 1.965570;...
    12.494170, 12.494170, 8.000000, 5.083520, 8.792390;...
    21.744240, 21.744240, 21.744240, 21.744240, 21.744240;...
    3.241680, 12.494170, 1.620760, 1.375250, 2.331620 ];

Coeffs(1,6,:,:)=     [...
    0.126970, 0.126970, 0.126970, 0.126970, 0.126970;...
    0.810820, 0.810820, 0.810820, 0.810820, 0.810820;...
    3.241680, 2.500000, 2.291440, 2.291440, 2.291440;...
    4.000000, 3.000000, 2.000000, 0.975430, 1.965570;...
    12.494170, 12.494170, 8.000000, 5.083520, 8.792390;...
    21.744240, 21.744240, 21.744240, 21.744240, 21.744240;...
    3.241680, 12.494170, 1.620760, 1.375250, 2.331620 ];

Coeffs(2,1,:,:)=     [...
    0.337440, 0.337440, 0.969110, 1.097190, 1.116080;...
    0.337440, 0.337440, 0.969110, 1.116030, 0.623900;...
    0.337440, 0.337440, 1.530590, 1.024420, 0.908480;...
    0.584040, 0.584040, 0.847250, 0.914940, 1.289300;...
    0.337440, 0.337440, 0.310240, 1.435020, 1.852830;...
    0.337440, 0.337440, 1.015010, 1.097190, 2.117230;...
    0.337440, 0.337440, 0.969110, 1.145730, 1.476400 ];

Coeffs(2,2,:,:)=     [...
    0.300000, 0.300000, 0.700000, 1.100000, 0.796940;...
    0.219870, 0.219870, 0.526530, 0.809610, 0.649300;...
    0.386650, 0.386650, 0.119320, 0.576120, 0.685460;...
    0.746730, 0.399830, 0.470970, 0.986530, 0.785370;...
    0.575420, 0.936700, 1.649200, 1.495840, 1.335590;...
    1.319670, 4.002570, 1.276390, 2.644550, 2.518670;...
    0.665190, 0.678910, 1.012360, 1.199940, 0.986580 ];

Coeffs(2,3,:,:)=     [...
    0.378870, 0.974060, 0.500000, 0.491880, 0.665290;...
    0.105210, 0.263470, 0.407040, 0.553460, 0.582590;...
    0.312900, 0.345240, 1.144180, 0.854790, 0.612280;...
    0.119070, 0.365120, 0.560520, 0.793720, 0.802600;...
    0.781610, 0.837390, 1.270420, 1.537980, 1.292950;...
    1.152290, 1.152290, 1.492080, 1.245370, 2.177100;...
    0.424660, 0.529550, 0.966910, 1.033460, 0.958730 ];

Coeffs(2,4,:,:)=     [...
    0.310590, 0.714410, 0.252450, 0.500000, 0.607600;...
    0.975190, 0.363420, 0.500000, 0.400000, 0.502800;...
    0.175580, 0.196250, 0.476360, 1.072470, 0.490510;...
    0.719280, 0.698620, 0.657770, 1.190840, 0.681110;...
    0.426240, 1.464840, 0.678550, 1.157730, 0.978430;...
    2.501120, 1.789130, 1.387090, 2.394180, 2.394180;...
    0.491640, 0.677610, 0.685610, 1.082400, 0.735410 ];

Coeffs(2,5,:,:)=     [...
    0.597000, 0.500000, 0.300000, 0.310050, 0.413510;...
    0.314790, 0.336310, 0.400000, 0.400000, 0.442460;...
    0.166510, 0.460440, 0.552570, 1.000000, 0.461610;...
    0.401020, 0.559110, 0.403630, 1.016710, 0.671490;...
    0.400360, 0.750830, 0.842640, 1.802600, 1.023830;...
    3.315300, 1.510380, 2.443650, 1.638820, 2.133990;...
    0.530790, 0.745850, 0.693050, 1.458040, 0.804500 ];

Coeffs(2,6,:,:)=     [...
    0.597000, 0.500000, 0.300000, 0.310050, 0.800920;...
    0.314790, 0.336310, 0.400000, 0.400000, 0.237040;...
    0.166510, 0.460440, 0.552570, 1.000000, 0.581990;...
    0.401020, 0.559110, 0.403630, 1.016710, 0.898570;...
    0.400360, 0.750830, 0.842640, 1.802600, 3.400390;...
    3.315300, 1.510380, 2.443650, 1.638820, 2.508780;...
    0.204340, 1.157740, 2.003080, 2.622080, 1.409380 ];

Coeffs(3,1,:,:)=     [...
    1.242210, 1.242210, 1.242210, 1.242210, 1.242210;...
    0.056980, 0.056980, 0.656990, 0.656990, 0.925160;...
    0.089090, 0.089090, 1.040430, 1.232480, 1.205300;...
    1.053850, 1.053850, 1.399690, 1.084640, 1.233340;...
    1.151540, 1.151540, 1.118290, 1.531640, 1.411840;...
    1.494980, 1.494980, 1.700000, 1.800810, 1.671600;...
    1.018450, 1.018450, 1.153600, 1.321890, 1.294670 ];

Coeffs(3,2,:,:)=     [...
    0.700000, 0.700000, 1.023460, 0.700000, 0.945830;...
    0.886300, 0.886300, 1.333620, 0.800000, 1.066620;...
    0.902180, 0.902180, 0.954330, 1.126690, 1.097310;...
    1.095300, 1.075060, 1.176490, 1.139470, 1.096110;...
    1.201660, 1.201660, 1.438200, 1.256280, 1.198060;...
    1.525850, 1.525850, 1.869160, 1.985410, 1.911590;...
    1.288220, 1.082810, 1.286370, 1.166170, 1.119330 ];

Coeffs(3,3,:,:)=     [...
    0.600000, 1.029910, 0.859890, 0.550000, 0.813600;...
    0.604450, 1.029910, 0.859890, 0.656700, 0.928840;...
    0.455850, 0.750580, 0.804930, 0.823000, 0.911000;...
    0.526580, 0.932310, 0.908620, 0.983520, 0.988090;...
    1.036110, 1.100690, 0.848380, 1.035270, 1.042380;...
    1.048440, 1.652720, 0.900000, 2.350410, 1.082950;...
    0.817410, 0.976160, 0.861300, 0.974780, 1.004580 ];

Coeffs(3,4,:,:)=     [...
    0.782110, 0.564280, 0.600000, 0.600000, 0.665740;...
    0.894480, 0.680730, 0.541990, 0.800000, 0.669140;...
    0.487460, 0.818950, 0.841830, 0.872540, 0.709040;...
    0.709310, 0.872780, 0.908480, 0.953290, 0.844350;...
    0.863920, 0.947770, 0.876220, 1.078750, 0.936910;...
    1.280350, 0.866720, 0.769790, 1.078750, 0.975130;...
    0.725420, 0.869970, 0.868810, 0.951190, 0.829220 ];

Coeffs(3,5,:,:)=     [...
    0.791750, 0.654040, 0.483170, 0.409000, 0.597180;...
    0.566140, 0.948990, 0.971820, 0.653570, 0.718550;...
    0.648710, 0.637730, 0.870510, 0.860600, 0.694300;...
    0.637630, 0.767610, 0.925670, 0.990310, 0.847670;...
    0.736380, 0.946060, 1.117590, 1.029340, 0.947020;...
    1.180970, 0.850000, 1.050000, 0.950000, 0.888580;...
    0.700560, 0.801440, 0.961970, 0.906140, 0.823880 ];

Coeffs(3,6,:,:) = [...
    0.500000, 0.500000, 0.586770, 0.470550, 0.629790;...
    0.500000, 0.500000, 1.056220, 1.260140, 0.658140;...
    0.500000, 0.500000, 0.631830, 0.842620, 0.582780;...
    0.554710, 0.734730, 0.985820, 0.915640, 0.898260;...
    0.712510, 1.205990, 0.909510, 1.078260, 0.885610;...
    1.899260, 1.559710, 1.000000, 1.150000, 1.120390;...
    0.653880, 0.793120, 0.903320, 0.944070, 0.796130 ];

Coeffs(4,1,:,:)=     [...
    1.000000, 1.000000, 1.050000, 1.170380, 1.178090;...
    0.960580, 0.960580, 1.059530, 1.179030, 1.131690;...
    0.871470, 0.871470, 0.995860, 1.141910, 1.114600;...
    1.201590, 1.201590, 0.993610, 1.109380, 1.126320;...
    1.065010, 1.065010, 0.828660, 0.939970, 1.017930;...
    1.065010, 1.065010, 0.623690, 1.119620, 1.132260;...
    1.071570, 1.071570, 0.958070, 1.114130, 1.127110 ];

Coeffs(4,2,:,:)=     [...
    0.950000, 0.973390, 0.852520, 1.092200, 1.096590;...
    0.804120, 0.913870, 0.980990, 1.094580, 1.042420;...
    0.737540, 0.935970, 0.999940, 1.056490, 1.050060;...
    1.032980, 1.034540, 0.968460, 1.032080, 1.015780;...
    0.900000, 0.977210, 0.945960, 1.008840, 0.969960;...
    0.600000, 0.750000, 0.750000, 0.844710, 0.899100;...
    0.926800, 0.965030, 0.968520, 1.044910, 1.032310 ];

Coeffs(4,3,:,:)=     [...
    0.850000, 1.029710, 0.961100, 1.055670, 1.009700;...
    0.818530, 0.960010, 0.996450, 1.081970, 1.036470;...
    0.765380, 0.953500, 0.948260, 1.052110, 1.000140;...
    0.775610, 0.909610, 0.927800, 0.987800, 0.952100;...
    1.000990, 0.881880, 0.875950, 0.949100, 0.893690;...
    0.902370, 0.875960, 0.807990, 0.942410, 0.917920;...
    0.856580, 0.928270, 0.946820, 1.032260, 0.972990 ];

Coeffs(4,4,:,:)=     [...
    0.750000, 0.857930, 0.983800, 1.056540, 0.980240;...
    0.750000, 0.987010, 1.013730, 1.133780, 1.038250;...
    0.800000, 0.947380, 1.012380, 1.091270, 0.999840;...
    0.800000, 0.914550, 0.908570, 0.999190, 0.915230;...
    0.778540, 0.800590, 0.799070, 0.902180, 0.851560;...
    0.680190, 0.317410, 0.507680, 0.388910, 0.646710;...
    0.794920, 0.912780, 0.960830, 1.057110, 0.947950 ];

Coeffs(4,5,:,:)=     [...
    0.750000, 0.833890, 0.867530, 1.059890, 0.932840;...
    0.979700, 0.971470, 0.995510, 1.068490, 1.030150;...
    0.858850, 0.987920, 1.043220, 1.108700, 1.044900;...
    0.802400, 0.955110, 0.911660, 1.045070, 0.944470;...
    0.884890, 0.766210, 0.885390, 0.859070, 0.818190;...
    0.615680, 0.700000, 0.850000, 0.624620, 0.669300;...
    0.835570, 0.946150, 0.977090, 1.049350, 0.979970 ];

Coeffs(4,6,:,:)=     [...
    0.689220, 0.809600, 0.900000, 0.789500, 0.853990;...
    0.854660, 0.852840, 0.938200, 0.923110, 0.955010;...
    0.938600, 0.932980, 1.010390, 1.043950, 1.041640;...
    0.843620, 0.981300, 0.951590, 0.946100, 0.966330;...
    0.694740, 0.814690, 0.572650, 0.400000, 0.726830;...
    0.211370, 0.671780, 0.416340, 0.297290, 0.498050;...
    0.843540, 0.882330, 0.911760, 0.898420, 0.960210 ];

Coeffs(5,1,:,:)=     [...
    1.054880, 1.075210, 1.068460, 1.153370, 1.069220;...
    1.000000, 1.062220, 1.013470, 1.088170, 1.046200;...
    0.885090, 0.993530, 0.942590, 1.054990, 1.012740;...
    0.920000, 0.950000, 0.978720, 1.020280, 0.984440;...
    0.850000, 0.908500, 0.839940, 0.985570, 0.962180;...
    0.800000, 0.800000, 0.810080, 0.950000, 0.961550;...
    1.038590, 1.063200, 1.034440, 1.112780, 1.037800 ];

Coeffs(5,2,:,:)=     [...
    1.017610, 1.028360, 1.058960, 1.133180, 1.045620;...
    0.920000, 0.998970, 1.033590, 1.089030, 1.022060;...
    0.912370, 0.949930, 0.979770, 1.020420, 0.981770;...
    0.847160, 0.935300, 0.930540, 0.955050, 0.946560;...
    0.880260, 0.867110, 0.874130, 0.972650, 0.883420;...
    0.627150, 0.627150, 0.700000, 0.774070, 0.845130;...
    0.973700, 1.006240, 1.026190, 1.071960, 1.017240 ];

Coeffs(5,3,:,:)=     [...
    1.028710, 1.017570, 1.025900, 1.081790, 1.024240;...
    0.924980, 0.985500, 1.014100, 1.092210, 0.999610;...
    0.828570, 0.934920, 0.994950, 1.024590, 0.949710;...
    0.900810, 0.901330, 0.928830, 0.979570, 0.913100;...
    0.761030, 0.845150, 0.805360, 0.936790, 0.853460;...
    0.626400, 0.546750, 0.730500, 0.850000, 0.689050;...
    0.957630, 0.985480, 0.991790, 1.050220, 0.987900 ];

Coeffs(5,4,:,:)=     [...
    0.992730, 0.993880, 1.017150, 1.059120, 1.017450;...
    0.975610, 0.987160, 1.026820, 1.075440, 1.007250;...
    0.871090, 0.933190, 0.974690, 0.979840, 0.952730;...
    0.828750, 0.868090, 0.834920, 0.905510, 0.871530;...
    0.781540, 0.782470, 0.767910, 0.764140, 0.795890;...
    0.743460, 0.693390, 0.514870, 0.630150, 0.715660;...
    0.934760, 0.957870, 0.959640, 0.972510, 0.981640 ];

Coeffs(5,5,:,:)=     [...
    0.965840, 0.941240, 0.987100, 1.022540, 1.011160;...
    0.988630, 0.994770, 0.976590, 0.950000, 1.034840;...
    0.958200, 1.018080, 0.974480, 0.920000, 0.989870;...
    0.811720, 0.869090, 0.812020, 0.850000, 0.821050;...
    0.682030, 0.679480, 0.632450, 0.746580, 0.738550;...
    0.668290, 0.445860, 0.500000, 0.678920, 0.696510;...
    0.926940, 0.953350, 0.959050, 0.876210, 0.991490 ];

Coeffs(5,6,:,:)=     [...
    0.948940, 0.997760, 0.850000, 0.826520, 0.998470;...
    1.017860, 0.970000, 0.850000, 0.700000, 0.988560;...
    1.000000, 0.950000, 0.850000, 0.606240, 0.947260;...
    1.000000, 0.746140, 0.751740, 0.598390, 0.725230;...
    0.922210, 0.500000, 0.376800, 0.517110, 0.548630;...
    0.500000, 0.450000, 0.429970, 0.404490, 0.539940;...
    0.960430, 0.881630, 0.775640, 0.596350, 0.937680 ];

Coeffs(6,1,:,:)=     [...
    1.030000, 1.040000, 1.000000, 1.000000, 1.049510;...
    1.050000, 0.990000, 0.990000, 0.950000, 0.996530;...
    1.050000, 0.990000, 0.990000, 0.820000, 0.971940;...
    1.050000, 0.790000, 0.880000, 0.820000, 0.951840;...
    1.000000, 0.530000, 0.440000, 0.710000, 0.928730;...
    0.540000, 0.470000, 0.500000, 0.550000, 0.773950;...
    1.038270, 0.920180, 0.910930, 0.821140, 1.034560 ];

Coeffs(6,2,:,:)=     [...
    1.041020, 0.997520, 0.961600, 1.000000, 1.035780;...
    0.948030, 0.980000, 0.900000, 0.950360, 0.977460;...
    0.950000, 0.977250, 0.869270, 0.800000, 0.951680;...
    0.951870, 0.850000, 0.748770, 0.700000, 0.883850;...
    0.900000, 0.823190, 0.727450, 0.600000, 0.839870;...
    0.850000, 0.805020, 0.692310, 0.500000, 0.788410;...
    1.010090, 0.895270, 0.773030, 0.816280, 1.011680 ];

Coeffs(6,3,:,:)=     [...
    1.022450, 1.004600, 0.983650, 1.000000, 1.032940;...
    0.943960, 0.999240, 0.983920, 0.905990, 0.978150;...
    0.936240, 0.946480, 0.850000, 0.850000, 0.930320;...
    0.816420, 0.885000, 0.644950, 0.817650, 0.865310;...
    0.742960, 0.765690, 0.561520, 0.700000, 0.827140;...
    0.643870, 0.596710, 0.474460, 0.600000, 0.651200;...
    0.971740, 0.940560, 0.714880, 0.864380, 1.001650 ];

Coeffs(6,4,:,:)=     [...
    0.995260, 0.977010, 1.000000, 1.000000, 1.035250;...
    0.939810, 0.975250, 0.939980, 0.950000, 0.982550;...
    0.876870, 0.879440, 0.850000, 0.900000, 0.917810;...
    0.873480, 0.873450, 0.751470, 0.850000, 0.863040;...
    0.761470, 0.702360, 0.638770, 0.750000, 0.783120;...
    0.734080, 0.650000, 0.600000, 0.650000, 0.715660;...
    0.942160, 0.919100, 0.770340, 0.731170, 0.995180 ];

Coeffs(6,5,:,:)=     [...
    0.952560, 0.916780, 0.920000, 0.900000, 1.005880;...
    0.928620, 0.994420, 0.900000, 0.900000, 0.983720;...
    0.913070, 0.850000, 0.850000, 0.800000, 0.924280;...
    0.868090, 0.807170, 0.823550, 0.600000, 0.844520;...
    0.769570, 0.719870, 0.650000, 0.550000, 0.733500;...
    0.580250, 0.650000, 0.600000, 0.500000, 0.628850;...
    0.904770, 0.852650, 0.708370, 0.493730, 0.949030 ];

Coeffs(6,6,:,:)=     [...
    0.911970, 0.800000, 0.800000, 0.800000, 0.956320;...
    0.912620, 0.682610, 0.750000, 0.700000, 0.950110;...
    0.653450, 0.659330, 0.700000, 0.600000, 0.856110;...
    0.648440, 0.600000, 0.641120, 0.500000, 0.695780;...
    0.570000, 0.550000, 0.598800, 0.400000, 0.560150;...
    0.475230, 0.500000, 0.518640, 0.339970, 0.520230;...
    0.743440, 0.592190, 0.603060, 0.316930, 0.794390 ];
%%

end